import sys
if "src.api_llm" in sys.modules:
    del sys.modules["src.api_llm"]

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import re
import datetime
import uuid
import google.generativeai as genai
from src.api_estat import search_stats_list, get_meta_info, get_stats_data
from src.data_processor import parse_estat_json_to_dataframe
from src.api_llm import chat_for_insights, chat_for_filtering, extract_json_parameters, generate_search_query, recommend_tables_from_list
from streamlit_local_storage import LocalStorage

# --- Version Info ---
APP_VERSION = "2026-04-19-1000"

# --- basic configurations ---
st.set_page_config(page_title=f"e-Stat AI Analyzer v{APP_VERSION}", layout="wide")
localS = LocalStorage()

# --- State Initialization ---
INIT_KEYS = {
    'estat_app_id': '',
    'gemini_api_key': '',
    'messages': [],
    'insight_messages': [],
    'current_df': None,
    'chat_mode': False,
    'filter_params': None,
    'readable_filter_summary': '',
    'selected_table_id_fixed': None,
    'selected_table_name': '',
    'meta_summary': '',
    'available_columns_details': [],
    'chart_config': {'chart_type': '折れ線', 'x_axis': None, 'y_axis': 'value', 'color_axis': None},
    'dimension_filters': {},
    'active_analysis_id': '', 
    'last_processed_id': ''
}
for k, v in INIT_KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if localS.storedItems is None:
    localS.storedItems = {}

@st.cache_resource
def get_global_gallery():
    return []

global_gallery = get_global_gallery()

ESTAT_CATEGORIES = {
    "01: 国土・気象": "01", "02: 人口・世帯": "02", "03: 労働・賃金": "03",
    "04: 農林水産業": "04", "05: 鉱工業": "05", "06: 商業・サービス業": "06",
    "07: 企業・家計・経済": "07", "08: 住宅・土地・建設": "08", "09: エネルギー・水": "09",
    "10: 運輸・観光": "10", "11: 情報通信・科学技術": "11",
    "12: 教育・文化・スポーツ・生活": "12", "13: 健康・医療": "13",
    "14: 福祉・社会保障": "14", "15: 行財政": "15", "16: 司法・安全・環境": "16", "17: その他": "17"
}

@st.cache_data(ttl=3600)
def fetch_gemini_models(api_key):
    if not api_key: return []
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
    except: return []

# --- helper for readable filters ---
def get_readable_filters(filter_params, meta_summary_json):
    if not filter_params or not meta_summary_json:
        return ""
    try:
        class_objs = json.loads(meta_summary_json)
        if isinstance(class_objs, dict): class_objs = [class_objs]
        readable = []
        for key, code in filter_params.items():
            dim_name = key
            val_name = code
            for obj in class_objs:
                if obj.get('@id') == key:
                    dim_name = obj.get('@name', key)
                    cl_list = obj.get('CLASS', [])
                    if isinstance(cl_list, dict): cl_list = [cl_list]
                    for cl in cl_list:
                        if str(cl.get('@code')) == str(code):
                            val_name = cl.get('@name', code)
                            break
                    break
            readable.append(f"・**{dim_name}**: {val_name}")
        return "\n".join(readable)
    except:
        return str(filter_params)

# --- core logic functions ---
def setup_analysis_phase(selected_table_id):
    if not st.session_state.get('estat_app_id') or not st.session_state.get('gemini_api_key'):
        st.error("API設定が不足しています。")
        return False
    with st.spinner("統計表のメタデータを取得中..."):
        try:
            meta_json = get_meta_info(selected_table_id, app_id=st.session_state['estat_app_id'])
            table_inf = meta_json.get('GET_META_INFO', {}).get('METADATA_INF', {}).get('TABLE_INF', {})
            # 統計表名の取得 (安全な抽出)
            title_item = table_inf.get('TITLE', table_inf.get('STAT_NAME', 'Unknown'))
            if isinstance(title_item, dict):
                st.session_state['selected_table_name'] = title_item.get('$', str(title_item))
            else:
                st.session_state['selected_table_name'] = str(title_item)
            raw = meta_json.get('GET_META_INFO', {}).get('METADATA_INF', {}).get('CLASS_INF', {})
            class_objs = raw.get('CLASS_OBJ', [])
            st.session_state['meta_summary'] = json.dumps(class_objs, ensure_ascii=False)
            if isinstance(class_objs, dict): class_objs = [class_objs]
            details = []
            for obj in class_objs:
                name = obj.get('@name', 'Unknown')
                cl_list = obj.get('CLASS', [])
                if isinstance(cl_list, dict): cl_list = [cl_list]
                ex = "、".join([str(c.get('@name','')) for c in cl_list[:3]])
                details.append(f"・**{name}** (例: {ex}{' など' if len(cl_list)>3 else ''})")
            st.session_state['available_columns_details'] = details
            st.session_state['selected_table_id_fixed'] = selected_table_id
            st.session_state['chat_mode'] = True
            st.session_state['active_analysis_id'] = str(uuid.uuid4())
            st.session_state['insight_messages'] = []
            st.session_state['filter_params'] = None
            st.session_state['readable_filter_summary'] = ''
            return True
        except Exception as e:
            st.error(f"統計表メタデータ取得エラー: {e}")
            return False

def restore_saved_analysis(item):
    app_id = st.session_state.get('estat_app_id')
    if not app_id: 
        st.error("API設定が必要です。")
        return False
    with st.spinner("🔄 統計表の分析レシピを復元中..."):
        try:
            st.session_state['selected_table_id_fixed'] = item['table_id']
            st.session_state['filter_params'] = item['filter_params']
            st.session_state['chart_config'] = item.get('chart_config', {'chart_type': '折れ線', 'x_axis': None, 'y_axis': 'value', 'color_axis': None})
            st.session_state['dimension_filters'] = item.get('dimension_filters', {})
            meta_json = get_meta_info(item['table_id'], app_id=app_id)
            table_inf = meta_json.get('GET_META_INFO', {}).get('METADATA_INF', {}).get('TABLE_INF', {})
            # 統計表名の取得 (安全な抽出)
            title_item = table_inf.get('TITLE', table_inf.get('STAT_NAME', 'Unknown'))
            if isinstance(title_item, dict):
                st.session_state['selected_table_name'] = title_item.get('$', str(title_item))
            else:
                st.session_state['selected_table_name'] = str(title_item)
            raw = meta_json.get('GET_META_INFO', {}).get('METADATA_INF', {}).get('CLASS_INF', {})
            class_objs = raw.get('CLASS_OBJ', [])
            if isinstance(class_objs, dict): class_objs = [class_objs]
            details = []
            for obj in class_objs:
                details.append(f"・**{obj.get('@name', 'Unknown')}**")
            st.session_state['available_columns_details'] = details
            st.session_state['meta_summary'] = json.dumps(class_objs, ensure_ascii=False)
            st.session_state['readable_filter_summary'] = get_readable_filters(item['filter_params'], st.session_state['meta_summary'])
            raw_json = get_stats_data(item['table_id'], app_id=app_id, filter_params=item['filter_params'])
            df = parse_estat_json_to_dataframe(raw_json)
            if df is not None:
                st.session_state['current_df'] = df
            st.session_state['active_analysis_id'] = str(uuid.uuid4())
            st.session_state['chat_mode'] = True
            for k in ['manual_tables', 'ai_recommendations', 'ai_search_query']:
                if k in st.session_state: del st.session_state[k]
            st.toast("✅ 統計表の分析環境を復元しました")
            return True
        except Exception as e:
            st.error(f"復元失敗: {e}")
            return False

# --- UI Setup ---
st.title(f"📊 e-Stat AI Analyzer (v{APP_VERSION})")
st.markdown("AIが政府統計の総合窓口(e-Stat)の**統計表の検索・絞り込み・可視化・インサイト分析**をサポートします。")

# --- Sidebar UI ---
if st.sidebar.button("🏠 トップに戻る (新規検索)", use_container_width=True, type="primary"):
    for k in list(st.session_state.keys()):
        if k not in ['estat_app_id', 'gemini_api_key']: del st.session_state[k]
    st.rerun()

with st.sidebar.expander("⚙️ API設定"):
    try:
        s_e = localS.getItem("estat_app_id"); s_g = localS.getItem("gemini_api_key")
    except: s_e = s_g = None
    if s_e and not st.session_state['estat_app_id']: st.session_state['estat_app_id'] = s_e
    if s_g and not st.session_state['gemini_api_key']: st.session_state['gemini_api_key'] = s_g
    c_e = st.text_input("e-Stat ID", value=st.session_state['estat_app_id'], type="password")
    c_g = st.text_input("Gemini API Key", value=st.session_state['gemini_api_key'], type="password")
    if st.button("設定を保存"):
        localS.setItem("estat_app_id", c_e); localS.setItem("gemini_api_key", c_g)
        st.session_state['estat_app_id'] = c_e; st.session_state['gemini_api_key'] = c_g; st.rerun()

models = fetch_gemini_models(st.session_state['gemini_api_key'])
llm_model = st.sidebar.selectbox("分析用AIモデル", models if models else ["gemini-1.5-flash"])

st.sidebar.divider()
is_admin = st.query_params.get("admin") == "true"
with st.sidebar.expander("🌍 共有ギャラリー", expanded=False):
    for i, item in enumerate(reversed(global_gallery)):
        if st.button(f"🔗 {item.get('title')}", key=f"g_btn_{i}"):
            if restore_saved_analysis(item): st.rerun()
    if is_admin:
        if st.button("🚨 全消去 (管理者)", key="admin_clear_gallery"): global_gallery.clear(); st.rerun()

with st.sidebar.expander("🔖 マイ・ブックマーク", expanded=True):
    try: bs = json.loads(localS.getItem("estat_my_bookmarks") or "[]")
    except: bs = []
    for i, item in enumerate(reversed(bs)):
        col_b, col_d = st.columns([4, 1])
        if col_b.button(f"📄 {item.get('title','無題')}", key=f"l_btn_{i}"):
            if restore_saved_analysis(item): st.rerun()
        if col_d.button("🗑️", key=f"ld_btn_{i}"):
            bs.pop(len(bs)-1-i); localS.setItem("estat_my_bookmarks", json.dumps(bs, ensure_ascii=False)); st.rerun()

# --- Search Phase ---
if not st.session_state.get('chat_mode', False):
    t1, t2 = st.tabs(["🤖 AIで統計表を探す", "📚 カテゴリから探す"])
    with t2:
        sc = st.selectbox("統計カテゴリ", list(ESTAT_CATEGORIES.keys()))
        kw = st.text_input("キーワード (任意)")
        if st.button("統計表を検索"):
            res = search_stats_list(ESTAT_CATEGORIES[sc], st.session_state['estat_app_id'], kw)
            if res: 
                st.session_state['manual_tables'] = res
                st.info(f"✅ {len(res)} 件の統計表が見つかりました。下のリストから選択してください。")
            else:
                st.warning("該当する統計表が見つかりませんでした。条件を変えてお試しください。")
        if 'manual_tables' in st.session_state:
            # 安全なタイトル抽出による選択肢の生成
            opts = {}
            for t in st.session_state['manual_tables']:
                t_item = t.get('TITLE', '無題')
                title = t_item.get('$', str(t_item)) if isinstance(t_item, dict) else str(t_item)
                opts[f"{title} ({t.get('@id', '')})"] = t.get('@id')
            sn = st.selectbox("対象の統計表を選択", list(opts.keys()))
            if st.button("分析を開始"):
                if setup_analysis_phase(opts[sn]): st.rerun()
    with t1:
        st.write("どんなデータを調べたいか入力してください")
        aq = st.text_input("検索テーマ", placeholder="例：最近の物価指数の傾向", key="ai_search_input")
        if st.button("AIで統計表を探索 🔍"):
            if aq:
                st.session_state['ai_search_query'] = aq
                with st.spinner("AIが最適な統計表を探索中..."):
                    p = generate_search_query(aq, ", ".join(ESTAT_CATEGORIES.keys()), st.session_state['gemini_api_key'], llm_model)
                    if p:
                        raw = search_stats_list(p['category_id'], st.session_state['estat_app_id'], p.get('search_keyword',''))
                        recs = recommend_tables_from_list(aq, raw, st.session_state['gemini_api_key'], llm_model)
                        if recs: st.session_state['ai_recommendations'] = recs
            else:
                st.warning("検索テーマを入力してください。")
        
        if 'ai_recommendations' in st.session_state:
            for rec in st.session_state['ai_recommendations']:
                with st.container(border=True):
                    # 🏅 -> 📊 に変更
                    st.subheader(f"📊 {rec.get('title')}")
                    st.caption(f"正式名称: {rec.get('stat_name')}")
                    st.write(f"**理由:** {rec.get('reason')}")
                    if st.button("分析を開始", key=f"ais_btn_{rec.get('id')}"):
                        if setup_analysis_phase(rec.get('id')): st.rerun()

# --- Analysis Phase ---
if st.session_state.get('chat_mode'):
    st.info(f"📍 **分析中:** {st.session_state.get('selected_table_name')} (ID: {st.session_state.get('selected_table_id_fixed')})")
    if st.session_state['active_analysis_id'] != st.session_state['last_processed_id']:
        st.session_state['last_processed_id'] = st.session_state['active_analysis_id']
        now = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state['insight_messages'] = [{"role": "user", "content": f"【解析リクエスト: {now}】データ解析をお願いします。"}]
        st.rerun()
    
    st.divider()
    st.subheader("💬 データの絞り込み相談")
    if st.session_state.get('available_columns_details'):
        with st.expander("📋 統計表の構成を見る", expanded=False):
            st.markdown("\n".join(st.session_state['available_columns_details']))
            
    for msg in st.session_state['messages']: st.chat_message(msg["role"]).write(msg["content"])
    p = st.chat_input("絞り込みの指示を入力（例：男性のみ）")
    if p:
        st.session_state['messages'].append({"role": "user", "content": p})
        with st.chat_message("assistant"):
            rep = chat_for_filtering(st.session_state['messages'], st.session_state['meta_summary'], st.session_state['gemini_api_key'], llm_model)
            st.session_state['messages'].append({"role": "assistant", "content": rep})
            ext = extract_json_parameters(rep)
            if ext: 
                st.session_state['filter_params'] = ext
                st.session_state['readable_filter_summary'] = get_readable_filters(ext, st.session_state['meta_summary'])
            st.rerun()
            
    if st.session_state.get('filter_params'):
        with st.container(border=True):
            st.write("**現在の絞り込み条件:**")
            st.markdown(st.session_state.get('readable_filter_summary'))
            if st.button("統計データを取得/更新 📊"):
                raw = get_stats_data(st.session_state['selected_table_id_fixed'], st.session_state['estat_app_id'], st.session_state['filter_params'])
                df = parse_estat_json_to_dataframe(raw)
                if df is not None: st.session_state['current_df'] = df; st.rerun()

if st.session_state.get('current_df') is not None:
    df_b = st.session_state['current_df']
    st.divider()
    st.subheader("📊 統計表の可視化設定")
    dims = [c for c in df_b.columns if c not in ['value', 'unit']]
    saved_config = st.session_state.get('chart_config', {})
    
    st.write("**詳細フィルタ（手動での絞り込み）:**")
    filters = {}
    if dims:
        fcols = st.columns(min(len(dims), 3))
        for i, d in enumerate(dims):
            uvs = [v for v in df_b[d].dropna().unique() if v != '']
            s_f = st.session_state.get('dimension_filters', {}).get(d)
            defs = [v for v in uvs if v in s_f] if s_f else [v for v in uvs if not any(k in str(v) for k in ['総数','計'])]
            filters[d] = fcols[i%3].multiselect(f"{d}の選択", uvs, default=defs or uvs)
    
    df_f = df_b.copy()
    for d, v in filters.items():
        if v: df_f = df_f[df_f[d].isin(v)]
    if df_f.empty and not df_b.empty: df_f = df_b.head(50)

    st.markdown("---")
    st.write("**グラフ構成:**")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    c_opts = ["折れ線", "棒", "円", "散布図"]
    with col_c1: ct = st.selectbox("グラフの種類", c_opts, index=c_opts.index(saved_config.get('chart_type', "折れ線")) if saved_config.get('chart_type') in c_opts else 0)
    with col_c2: x_axis = st.selectbox("X軸 (横軸)", dims, index=dims.index(saved_config.get('x_axis')) if saved_config.get('x_axis') in dims else 0)
    y_opts = ['value'] + dims
    with col_c3: y_axis = st.selectbox("Y軸 (縦軸)", y_opts, index=y_opts.index(saved_config.get('y_axis', 'value')) if saved_config.get('y_axis', 'value') in y_opts else 0)
    available_color_axes = [None] + dims
    c_idx = available_color_axes.index(saved_config.get('color_axis')) if saved_config.get('color_axis') in available_color_axes else 0
    with col_c4: color_axis = st.selectbox("色分け (凡例)", available_color_axes, index=c_idx)
    
    if not df_f.empty:
        # X軸の値で昇順ソート（時系列順にするための処理）
        df_f = df_f.sort_values(by=x_axis)
        
        st.info(f"💡 現在の表示対象データ: {len(df_f)} 件")
        p_params = {"data_frame": df_f, "x": x_axis, "y": y_axis, "color": color_axis}
        if ct == "棒": fig = px.bar(**p_params)
        elif ct == "円": fig = px.pie(df_f, names=x_axis, values=y_axis, color=color_axis)
        elif ct == "散布図": fig = px.scatter(**p_params)
        else: fig = px.line(**p_params)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("💡 AIアナリストの解析レポート")
    if st.button("🔄 この状態で再解析"): st.session_state['last_processed_id'] = ""; st.rerun()
    for m in st.session_state.get('insight_messages', []): st.chat_message(m["role"]).write(m["content"])
    if st.session_state.get('insight_messages') and st.session_state['insight_messages'][-1]["role"] == "user":
        with st.status("AI解析中...") as status:
            summary = f"表示データ: {len(df_f)}件\nX={x_axis}, Y={y_axis}, 色={color_axis}\nサンプル:\n{df_f.head(10).to_string()}"
            res = chat_for_insights(st.session_state['insight_messages'], summary, st.session_state['gemini_api_key'], llm_model)
            st.session_state['insight_messages'].append({"role": "assistant", "content": res})
            status.update(label="✅ 解析完了", state="complete")
            st.rerun()

    st.divider()
    t_input = st.text_input("保存タイトル", value=f"分析: {datetime.datetime.now().strftime('%m/%d %H:%M')}")
    col_s1, col_s2 = st.columns(2)
    curr_cfg = {"chart_type": ct, "x_axis": x_axis, "y_axis": y_axis, "color_axis": color_axis}
    if col_s1.button("💾 マイ・ブックマークに保存"):
        item = {
            "title": t_input, "table_id": st.session_state['selected_table_id_fixed'],
            "filter_params": st.session_state['filter_params'], "insight_messages": [], 
            "chart_config": curr_cfg, "dimension_filters": filters, "timestamp": datetime.datetime.now().strftime("%m/%d %H:%M")
        }
        l = json.loads(localS.getItem("estat_my_bookmarks") or "[]")
        l.append(item); localS.setItem("estat_my_bookmarks", json.dumps(l, ensure_ascii=False)); st.rerun()
    if col_s2.button("🌍 全体ギャラリーに共有"):
        item = {
            "title": t_input, "table_id": st.session_state['selected_table_id_fixed'],
            "filter_params": st.session_state['filter_params'], "insight_messages": [], 
            "chart_config": curr_cfg, "dimension_filters": filters, "timestamp": datetime.datetime.now().strftime("%m/%d %H:%M")
        }
        global_gallery.append(item); st.rerun()
