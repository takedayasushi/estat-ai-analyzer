import sys
# Streamlit Cloudの強力すぎるモジュールキャッシュを強制削除するハック（ImportError対策）
if "src.api_llm" in sys.modules:
    del sys.modules["src.api_llm"]

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import re
import google.generativeai as genai
from src.api_estat import search_stats_list, get_meta_info, get_stats_data
from src.data_processor import parse_estat_json_to_dataframe
from src.api_llm import chat_for_insights, chat_for_filtering, extract_json_parameters, generate_search_query, recommend_tables_from_list
from streamlit_local_storage import LocalStorage

# --- basic configurations ---
st.set_page_config(page_title="e-Stat AI Analyzer", layout="wide")
localS = LocalStorage()
# st_local_storageのバグ対策：初期化時にNoneになる問題を回避
if localS.storedItems is None:
    localS.storedItems = {}

@st.cache_resource
def get_global_gallery():
    """全ユーザーで共有されるインメモリのリスト"""
    return []

global_gallery = get_global_gallery()

ESTAT_CATEGORIES = {
    "01: 国土・気象": "01",
    "02: 人口・世帯": "02",
    "03: 労働・賃金": "03",
    "04: 農林水産業": "04",
    "05: 鉱工業": "05",
    "06: 商業・サービス業": "06",
    "07: 企業・家計・経済": "07",
    "08: 住宅・土地・建設": "08",
    "09: エネルギー・水": "09",
    "10: 運輸・観光": "10",
    "11: 情報通信・科学技術": "11",
    "12: 教育・文化・スポーツ・生活": "12",
    "13: 健康・医療": "13",
    "14: 福祉・社会保障": "14",
    "15: 行財政": "15",
    "16: 司法・安全・環境": "16",
    "17: その他": "17"
}

@st.cache_data(ttl=3600)
def fetch_gemini_models(api_key):
    if not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        valid_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                if re.search(r'gemini(-[0-9.]+)?-(pro|flash|ultra|lite)', name.lower()):
                    if not any(bad in name.lower() for bad in ['vision', 'exp', 'image', 'audio', 'video', 'embed']):
                        valid_models.append(name)
        return valid_models
    except:
        return []

# --- app global states ---
app_id_ready = bool(st.session_state.get('estat_app_id'))
api_key_ready = bool(st.session_state.get('gemini_api_key'))

# --- core logic functions ---
def setup_analysis_phase(selected_table_id):
    if not app_id_ready or not api_key_ready:
        st.error("e-Stat IDとGemini API Keyの両方が必要です。")
        return False
    with st.spinner("表の構造（メタデータ）を読み込んでいます..."):
        try:
            meta_json = get_meta_info(selected_table_id, app_id=st.session_state['estat_app_id'])
            class_objs = meta_json.get('GET_META_INFO', {}).get('METADATA_INF', {}).get('CLASS_INF', {}).get('CLASS_OBJ', [])
            meta_summary = json.dumps(class_objs, ensure_ascii=False)
            
            st.session_state['meta_summary'] = meta_summary
            st.session_state['selected_table_id_fixed'] = selected_table_id
            st.session_state['chat_mode'] = True
            
            if isinstance(class_objs, dict):
                class_objs = [class_objs]
                
            col_details = []
            for obj in class_objs:
                col_name = obj.get('@name', obj.get('@id', 'Unknown'))
                classes = obj.get('CLASS', [])
                if isinstance(classes, dict): classes = [classes]
                examples = [str(cls.get('@name', cls.get('@code', ''))) for cls in classes[:3]]
                ex_str = "、".join(examples)
                if len(classes) > 3: ex_str += " など"
                col_details.append(f"・**{col_name}**{'（例: ' + ex_str + '）' if ex_str else ''}")
                    
            st.session_state['available_columns_details'] = col_details
            
            details_str = "\n".join(col_details)
            st.session_state['messages'] = [
                {"role": "assistant", "content": f"データ構造を読み込みました！このデータには以下の項目（カラム）が含まれています。\n\n{details_str}\n\nどの項目を、どのような条件（例：東京都のみ、など）で絞り込みたいかチャットで指示してください！"}
            ]
            st.session_state['filter_params'] = None
            return True
        except Exception as e:
            st.error(f"メタデータ取得エラー: {e}")
            return False

def restore_saved_analysis(item):
    """保存されたデータから分析環境を完全に復元する"""
    if not app_id_ready or not api_key_ready:
        return False
        
    # 旧形式または不完全なデータのチェック
    required_keys = ['table_id', 'filter_params', 'insight_messages']
    if not isinstance(item, dict) or not all(k in item for k in required_keys):
        st.error("⚠️ この保存データは形式が古いため、現在のバージョンでは復元できません。")
        return False
        
    with st.status("🔄 保存時のレシピから分析環境を再現中...", expanded=True) as status:
        try:
            # 1. 基本変数の復旧
            t_id = item['table_id']
            f_params = item['filter_params']
            st.session_state['selected_table_id_fixed'] = t_id
            st.session_state['filter_params'] = f_params
            st.session_state['insight_messages'] = item['insight_messages']
            st.session_state['chat_mode'] = True
            
            # 検索ステートのクリア
            for k in ['manual_tables', 'ai_recommendations', 'ai_search_query']:
                if k in st.session_state: del st.session_state[k]
                
            # 2. メタデータの再取得
            st.write("・e-Statから表の構造情報を取得しています...")
            meta_json = get_meta_info(t_id, app_id=st.session_state['estat_app_id'])
            class_objs = meta_json.get('GET_META_INFO', {}).get('METADATA_INF', {}).get('CLASS_INF', {}).get('CLASS_OBJ', [])
            st.session_state['meta_summary'] = json.dumps(class_objs, ensure_ascii=False)
            
            if isinstance(class_objs, dict): class_objs = [class_objs]
            col_details = []
            for obj in class_objs:
                col_name = obj.get('@name', obj.get('@id', 'Unknown'))
                classes = obj.get('CLASS', [])
                if isinstance(classes, dict): classes = [classes]
                examples = [str(cls.get('@name', cls.get('@code', ''))) for cls in classes[:3]]
                ex_str = "、".join(examples)
                if len(classes) > 3: ex_str += " など"
                col_details.append(f"・**{col_name}**{'（例: ' + ex_str + '）' if ex_str else ''}")
            st.session_state['available_columns_details'] = col_details
            
            # 3. 実データの再取得
            st.write("・保存時の条件でe-Stat APIから最新データを取得しています...")
            raw_json = get_stats_data(t_id, app_id=st.session_state['estat_app_id'], filter_params=f_params)
            df = parse_estat_json_to_dataframe(raw_json)
            if df is not None and not df.empty:
                st.session_state['current_df'] = df
                st.write("✅ データの取得に成功しました。")
            else:
                st.write("⚠️ データの取得に失敗したか、0件でした。")
                
            # 4. 追加：グラフ設定とフィルタの復元
            if 'chart_config' in item:
                st.session_state['chart_config'] = item['chart_config']
            if 'dimension_filters' in item:
                st.session_state['dimension_filters'] = item['dimension_filters']

            status.update(label="✅ 再現完了！最新データでグラフと考察を復元しました。", state="complete")
            return True
        except Exception as e:
            status.update(label=f"❌ 復元エラー: {e}", state="error")
            return False

# --- UI Layout ---
st.title("📊 e-Stat AI Analyzer")
st.markdown("""
**政府統計の総合窓口「e-Stat」の巨大で複雑なデータを、誰もが簡単に可視化・分析できるエージェント型ツールです。**  
1. **探す**: カテゴリとキーワードから目的の統計表を検索。
2. **絞る**: AIがデータ構造を自動解読。チャットで指示するだけで狙ったデータを抽出。
3. **描画・分析**: グラフとAIによる考察インサイトを瞬時に生成。
""")

# --- Settings Sidebar ---
with st.sidebar.expander("⚙️ API設定", expanded=not app_id_ready):
    # LocalStorageから読み込み (初期化)
    try:
        saved_estat = localS.getItem("estat_app_id")
        saved_gemini = localS.getItem("gemini_api_key")
    except:
        saved_estat = saved_gemini = None

    if saved_estat and 'estat_app_id' not in st.session_state: st.session_state['estat_app_id'] = saved_estat
    if saved_gemini and 'gemini_api_key' not in st.session_state: st.session_state['gemini_api_key'] = saved_gemini

    current_estat = st.text_input("e-Stat Application ID", value=st.session_state.get('estat_app_id', ""), type="password")
    current_gemini = st.text_input("Gemini API Key", value=st.session_state.get('gemini_api_key', ""), type="password")
    
    if st.button("設定をブラウザに保存"):
        localS.setItem("estat_app_id", current_estat, key="set_estat")
        localS.setItem("gemini_api_key", current_gemini, key="set_gemini")
        st.session_state['estat_app_id'] = current_estat
        st.session_state['gemini_api_key'] = current_gemini
        fetch_gemini_models.clear()
        st.success("設定を保存しました。")
        st.rerun()

available_models = fetch_gemini_models(st.session_state.get('gemini_api_key'))
if not available_models:
    llm_model = st.sidebar.selectbox("クラウドAIモデル", ["APIキーを設定して下さい"], disabled=True)
else:
    default_index = 0
    for i, m in enumerate(available_models):
        if "gemini-1.5-flash" in m:
            default_index = i
            break
    llm_model = st.sidebar.selectbox("クラウドAIモデル", available_models, index=default_index)
    st.session_state['llm_model'] = llm_model

# --- Gallery & Bookmarks Sidebar ---
st.sidebar.divider()

# 管理者モードのチェック (?admin=true)
is_admin = st.query_params.get("admin") == "true"

with st.sidebar.expander("🌍 全体ギャラリー", expanded=False):
    st.caption("※サーバー再起動でリセットされます。復元時はAPIで再取得するため最新の状態が反映されます。")
    if not global_gallery:
        st.caption("現在、共有されている分析はありません。")
    else:
        for i, item in enumerate(reversed(global_gallery)):
            if st.button(f"🔗 {item.get('title', 'Unknown')} ({item.get('timestamp', '')})", key=f"global_load_{i}"):
                if restore_saved_analysis(item):
                    st.rerun()
                else:
                    st.error("復元に失敗しました。")
    
    # 管理者専用の削除ボタン (URLに ?admin=true がある時のみ表示)
    if is_admin:
        st.markdown("---")
        if st.button("🚨 全体ギャラリーを全削除 (管理者)", key="admin_clear_gallery"):
            global_gallery.clear()
            st.rerun()
                
with st.sidebar.expander("🔖 マイ・ブックマーク", expanded=False):
    st.caption("※お使いのブラウザに保存された条件レシピです。")
    try:
        saved_str = localS.getItem("estat_my_bookmarks")
        my_bookmarks = json.loads(saved_str) if saved_str else []
    except:
        my_bookmarks = []
        
    if not my_bookmarks:
        st.caption("保存済みの分析はありません。")
    else:
        for i, item in enumerate(reversed(my_bookmarks)):
            col_b, col_d = st.columns([4, 1])
            if col_b.button(f"📄 {item.get('title', '無題')}", key=f"local_load_{i}"):
                if restore_saved_analysis(item):
                    st.rerun()
            if col_d.button("🗑️", key=f"local_del_{i}"):
                my_bookmarks.pop(len(my_bookmarks) - 1 - i)
                localS.setItem("estat_my_bookmarks", json.dumps(my_bookmarks, ensure_ascii=False), key=f"del_{i}")
                st.rerun()
    
    if st.button("🗑️ ブックマークを全消去", key="clear_all_bookmarks"):
        localS.setItem("estat_my_bookmarks", "[]", key="clear_all_bookmarks_trigger")
        st.rerun()

# --- Main Tabs ---
st.divider()
tab_ai, tab_manual = st.tabs(["🤖 AIにおまかせ検索", "📚 手動カテゴリ検索"])

with tab_manual:
    st.markdown("### 📚 カテゴリから探す")
    selected_cat_name = st.selectbox("検索カテゴリ", list(ESTAT_CATEGORIES.keys()))
    stats_field_code = ESTAT_CATEGORIES[selected_cat_name]
    search_keyword = st.text_input("キーワード入力", value="", placeholder="例: 消費支出, 雇用")
    
    if st.button("統計表を検索"):
        if not app_id_ready:
            st.error("e-Stat Application IDを設定してください。")
        else:
            with st.spinner("e-Statを検索中..."):
                try:
                    tables = search_stats_list(stats_field=stats_field_code, app_id=st.session_state['estat_app_id'], search_word=search_keyword)
                    if tables:
                        st.session_state['manual_tables'] = tables
                        for k in ['chat_mode', 'filter_params', 'current_df', 'insight_messages', 'selected_table_id_fixed', 'ai_recommendations', 'ai_search_query', 'last_query']:
                            if k in st.session_state: del st.session_state[k]
                        st.success(f"{len(tables)}件見つかりました。")
                    else:
                        st.warning("見つかりませんでした。")
                except Exception as e:
                    st.error(f"エラー: {e}")
                    
    if 'manual_tables' in st.session_state:
        def format_t(t): return f"{t.get('TITLE', {}).get('$', t.get('TITLE', ''))}"
        table_options = {format_t(t): t.get('@id', '') for t in st.session_state['manual_tables']}
        sel_name = st.selectbox("統計表を選択", list(table_options.keys()))
        sel_id = table_options[sel_name]
        if st.button("この表で分析を開始"):
            if setup_analysis_phase(sel_id): st.rerun()

with tab_ai:
    st.markdown("### 🤖 コンシェルジュに相談")
    if st.session_state.get('chat_mode'):
        st.warning("現在、選択した表の分析中です。別の検索をするにはリセットしてください。")
        if st.button("検索画面に戻る（リセット）"):
            for k in ['chat_mode', 'manual_tables', 'ai_recommendations', 'ai_search_query', 'current_df', 'filter_params', 'insight_messages', 'selected_table_id_fixed']:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
    else:
        ai_q = st.chat_input("何について調べたいですか？")
        if ai_q:
            if not api_key_ready or not app_id_ready:
                st.error("API設定が不足しています。")
            else:
                st.session_state['ai_search_query'] = ai_q
                for k in ['manual_tables', 'chat_mode', 'current_df', 'filter_params', 'insight_messages', 'selected_table_id_fixed']:
                    if k in st.session_state: del st.session_state[k]

    if 'ai_search_query' in st.session_state:
        q = st.session_state['ai_search_query']
        st.chat_message("user").write(q)
        if 'ai_recommendations' not in st.session_state or st.session_state.get('last_query') != q:
            st.session_state['last_query'] = q
            with st.status("AIが検索条件を考えています...", expanded=True) as status:
                params = generate_search_query(q, ", ".join(ESTAT_CATEGORIES.keys()), st.session_state['gemini_api_key'], llm_model)
                if params:
                    raw = search_stats_list(params['category_id'], st.session_state['estat_app_id'], params.get('search_keyword', ''))
                    if raw:
                        recs = recommend_tables_from_list(q, raw, st.session_state['gemini_api_key'], llm_model)
                        if recs:
                            st.session_state['ai_recommendations'] = recs
                            st.session_state['raw_tables_dict'] = {t.get('@id'): t for t in raw}
                            status.update(label="おすすめを見つけました！", state="complete")
        
        if 'ai_recommendations' in st.session_state:
            for rec in st.session_state['ai_recommendations']:
                with st.container(border=True):
                    st.subheader(f"🏅 {rec.get('title', rec.get('id', '不明'))}")
                    st.info(rec.get('reason', ''))
                    if st.button("この表で分析を開始", key=f"ai_sel_{rec.get('id')}"):
                        if setup_analysis_phase(rec.get('id')): st.rerun()

# --- Analysis Phase ---
if st.session_state.get('chat_mode'):
    st.divider()
    if 'messages' not in st.session_state: st.session_state['messages'] = []
    if 'insight_messages' not in st.session_state: st.session_state['insight_messages'] = []
    
    st.subheader("💬 データの絞り込み相談")
    if 'available_columns_details' in st.session_state:
        st.info("\n".join(st.session_state['available_columns_details']))
        
    for msg in st.session_state['messages']: st.chat_message(msg["role"]).write(msg["content"])
    
    prompt = st.chat_input("例：東京都だけに絞って、最新の5年間分を表示して")
    if prompt:
        st.session_state['filter_params'] = None
        if 'current_df' in st.session_state: del st.session_state['current_df']
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            reply = chat_for_filtering(st.session_state['messages'], st.session_state['meta_summary'], st.session_state['gemini_api_key'], llm_model)
            st.write(reply)
            st.session_state['messages'].append({"role": "assistant", "content": reply})
            extracted = extract_json_parameters(reply)
            if extracted: st.session_state['filter_params'] = extracted
            st.rerun()

    if st.session_state.get('filter_params'):
        if st.button("データを取得して可視化"):
            try:
                raw = get_stats_data(st.session_state['selected_table_id_fixed'], st.session_state['estat_app_id'], st.session_state['filter_params'])
                df = parse_estat_json_to_dataframe(raw)
                if df is not None and not df.empty:
                    st.session_state['current_df'] = df
                    st.success("成功！")
                else: st.warning("0件です。")
            except Exception as e: st.error(f"取得失敗: {e}")

if st.session_state.get('current_df') is not None:
    df_base = st.session_state['current_df']
    st.subheader("📊 データの確認と可視化")
    dims = [c for c in df_base.columns if c not in ['value', 'unit']]
    filters = {}
    if dims:
        cols = st.columns(min(len(dims), 4))
        for i, d in enumerate(dims):
            uvs = [v for v in df_base[d].dropna().unique() if v != '']
            
            # 復元されたフィルタ値がある場合はそれを優先、なければデフォルト（総数以外）
            saved_filters = st.session_state.get('dimension_filters', {}).get(d)
            if saved_filters:
                defaults = [v for v in uvs if v in saved_filters]
            else:
                defaults = [v for v in uvs if not any(k in str(v) for k in ['総数', '計', '不詳'])]
            
            filters[d] = cols[i%4].multiselect(d, uvs, default=defaults or uvs)
    
    df_f = df_base.copy()
    for d, v in filters.items():
        if v: df_f = df_f[df_f[d].isin(v)]
    
    if not df_f.empty:
        st.dataframe(df_f.head())
        col_c1, col_c2, col_c3 = st.columns(3)
        
        # グラフ種類の初期値
        chart_type_options = ["折れ線", "棒", "円"]
        saved_ct = st.session_state.get('chart_config', {}).get('chart_type')
        ct_idx = chart_type_options.index(saved_ct) if saved_ct in chart_type_options else 0
        ct = col_c1.selectbox("グラフ", chart_type_options, index=ct_idx)
        
        # X軸の初期値
        saved_xa = st.session_state.get('chart_config', {}).get('x_axis')
        xa_idx = dims.index(saved_xa) if saved_xa in dims else 0
        xa = col_c2.selectbox("X軸", dims, index=xa_idx)
        
        # 色分けの初期値
        cc_options = ["なし"] + [d for d in dims if d != xa]
        saved_cc = st.session_state.get('chart_config', {}).get('color_col')
        cc_idx = cc_options.index(saved_cc) if saved_cc in cc_options else 0
        cc = col_c3.selectbox("色分け", cc_options, index=cc_idx)
        
        fig = None
        if ct == "折れ線": fig = px.line(df_f, x=xa, y='value', color=cc if cc != "なし" else None, markers=True)
        elif ct == "棒": fig = px.bar(df_f, x=xa, y='value', color=cc if cc != "なし" else None, barmode='group')
        else: fig = px.pie(df_f, names=xa, values='value')
        
        if fig: st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("💡 AIアナリストの考察")
        if not st.session_state['insight_messages']:
            if st.button("初期レポートを生成"):
                st.session_state['insight_messages'].append({"role": "user", "content": "このデータの傾向を分析して"})
                st.rerun()
        
        for m in st.session_state['insight_messages']: st.chat_message(m["role"]).write(m["content"])
        
        if st.session_state['insight_messages'] and st.session_state['insight_messages'][-1]["role"] == "user":
            with st.chat_message("assistant"):
                summary = f"データ数: {len(df_f)}\n内容:\n{df_f.head(5).to_string()}"
                res = chat_for_insights(st.session_state['insight_messages'], summary, st.session_state['gemini_api_key'], llm_model)
                st.write(res)
                st.session_state['insight_messages'].append({"role": "assistant", "content": res})
                st.rerun()

        with st.form("qa_form"):
            qa_q = st.text_input("追加で質問")
            if st.form_submit_button("送信"):
                st.session_state['insight_messages'].append({"role": "user", "content": qa_q})
                st.rerun()

        st.divider()
        st.subheader("💾 保存・公開")
        c_s1, c_s2 = st.columns(2)
        
        # 保存時点で現在の設定をパッケージング
        chart_config = {
            "chart_type": ct,
            "x_axis": xa,
            "color_col": cc
        }
        
        with c_s1:
            t_s = st.text_input("保存名", key="local_title")
            if st.button("ブラウザに保存"):
                item = {
                    "title": t_s,
                    "table_id": st.session_state['selected_table_id_fixed'],
                    "filter_params": st.session_state['filter_params'],
                    "insight_messages": st.session_state['insight_messages'],
                    "chart_config": chart_config,
                    "dimension_filters": filters,
                    "timestamp": pd.Timestamp.now().strftime("%m/%d %H:%M")
                }
                l = json.loads(localS.getItem("estat_my_bookmarks") or "[]")
                l.append(item)
                localS.setItem("estat_my_bookmarks", json.dumps(l, ensure_ascii=False))
                st.success("保存完了")
        with c_s2:
            t_g = st.text_input("公開名", key="global_title")
            if st.button("全体に共有"):
                item = {
                    "title": t_g,
                    "table_id": st.session_state['selected_table_id_fixed'],
                    "filter_params": st.session_state['filter_params'],
                    "insight_messages": st.session_state['insight_messages'],
                    "chart_config": chart_config,
                    "dimension_filters": filters,
                    "timestamp": pd.Timestamp.now().strftime("%m/%d %H:%M")
                }
                global_gallery.append(item)
                st.success("共有完了")
