import streamlit as st
import pandas as pd
import plotly.express as px
import json
import google.generativeai as genai
from src.api_estat import search_stats_list, get_meta_info, get_stats_data
from src.data_processor import parse_estat_json_to_dataframe
from src.api_llm import generate_insights, chat_for_filtering, extract_json_parameters
from streamlit_local_storage import LocalStorage

st.set_page_config(page_title="e-Stat AI Analyzer", layout="wide")
localS = LocalStorage()

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
def get_gemini_models(api_key):
    if not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        # Fetch available models supporting text generation, filtering out non-text/specialized models
        valid_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                # 画像専用、埋め込み、音声などの非チャットモデルを除外
                if any(x in name.lower() for x in ['embedding', 'aqa', 'vision', 'imagen', 'audio']):
                    continue
                # Geminiシリーズに限定
                if 'gemini' in name.lower():
                    valid_models.append(name)
        return valid_models
    except:
        return []

st.title("📊 e-Stat AI Analyzer")
st.markdown("""
**政府統計の総合窓口「e-Stat」の巨大で複雑なデータを、誰もが簡単に可視化・分析できるエージェント型ツールです。**  
1. **探す**: カテゴリとキーワードから目的の統計表を検索します。
2. **絞る**: AI（Gemini）とチャットするだけで、複雑なデータ構造（地域・年齢など）をAIが自動解読して狙ったデータを抽出します。
3. **描画・分析**: ノイズ（総数等）をワンタッチで除外し、美しいグラフとAIによる考察インサイトを一瞬で生成します。
""")

# --- Settings Sidebar ---
with st.sidebar.expander("⚙️ API設定", expanded=True):
    saved_estat = localS.getItem("estat_app_id")
    saved_gemini = localS.getItem("gemini_api_key")

    if 'estat_app_id' not in st.session_state:
        st.session_state['estat_app_id'] = ""
    if 'gemini_api_key' not in st.session_state:
        st.session_state['gemini_api_key'] = ""

    # LocalStorageから読み込めた場合、セッションステートへ代入
    if saved_estat and isinstance(saved_estat, str):
        st.session_state['estat_app_id'] = saved_estat
    if saved_gemini and isinstance(saved_gemini, str):
        st.session_state['gemini_api_key'] = saved_gemini

    current_estat = st.text_input("e-Stat Application ID", value=st.session_state['estat_app_id'], type="password")
    current_gemini = st.text_input("Gemini API Key", value=st.session_state['gemini_api_key'], type="password")
    
    if st.button("設定をブラウザに保存"):
        localS.setItem("estat_app_id", current_estat, key="set_estat")
        localS.setItem("gemini_api_key", current_gemini, key="set_gemini")
        st.session_state['estat_app_id'] = current_estat
        st.session_state['gemini_api_key'] = current_gemini
        # APIキーが変わったらモデル一覧を再取得するためにキャッシュクリア
        get_gemini_models.clear()
        st.success("ブラウザ(LocalStorage)に安全に保存しました。")

st.sidebar.header("検索設定")
selected_cat_name = st.sidebar.selectbox("検索カテゴリ（大分類）", list(ESTAT_CATEGORIES.keys()))
stats_field_code = ESTAT_CATEGORIES[selected_cat_name]
search_keyword = st.sidebar.text_input("さらに絞り込むキーワード（任意）", value="", placeholder="例: 出生率, 女性など")

available_models = get_gemini_models(st.session_state.get('gemini_api_key'))
if not available_models:
    llm_model = st.sidebar.selectbox("クラウドAIモデル", ["APIキーを設定して下さい"], disabled=True)
else:
    default_index = 0
    # なるべく最新のflashモデルをデフォルトにする
    for i, m in enumerate(available_models):
        if "gemini-1.5-flash" in m or "gemini-2.5-flash" in m:
            default_index = i
            break
    llm_model = st.sidebar.selectbox("クラウドAIモデル", available_models, index=default_index)

app_id_ready = bool(st.session_state.get('estat_app_id'))
api_key_ready = bool(st.session_state.get('gemini_api_key'))

if st.sidebar.button("統計表を検索"):
    if not app_id_ready:
        st.sidebar.error("先に「⚙️ API設定」からe-Stat IDを保存してください。")
    else:
        with st.spinner("e-Statを検索中..."):
            try:
                tables = search_stats_list(stats_field=stats_field_code, app_id=st.session_state['estat_app_id'], search_word=search_keyword)
                if not tables:
                    st.warning("指定された条件で見つかりませんでした。別のキーワードなどで試してください。")
                else:
                    st.session_state['tables'] = tables
                    # Reset states
                    if 'chat_mode' in st.session_state: del st.session_state['chat_mode']
                    if 'filter_params' in st.session_state: del st.session_state['filter_params']
                    if 'current_df' in st.session_state: del st.session_state['current_df']
                    st.success(f"{len(tables)} 件の統計表が見つかりました！")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

# 結果があれば選択肢として表示
if 'tables' in st.session_state:
    def format_table_name(t):
        if not isinstance(t, dict):
            return "Unknown Table"
        stat_name_obj = t.get('STAT_NAME', '')
        stat_name = stat_name_obj.get('$', '') if isinstance(stat_name_obj, dict) else str(stat_name_obj)
        title_obj = t.get('TITLE', '')
        title = title_obj.get('$', '') if isinstance(title_obj, dict) else str(title_obj)
        title_no = title_obj.get('@no', '') if isinstance(title_obj, dict) else ''
        prefix = f"{title_no} " if title_no else ""
        return f"{prefix}{stat_name} - {title}"

    table_options = {format_table_name(t): t.get('@id', '') for t in st.session_state['tables'] if isinstance(t, dict)}
    
    selected_table_name = st.selectbox("分析する統計表を選択してください", list(table_options.keys()))
    selected_table_id = table_options[selected_table_name]
    
    # 選択した表の概要（メタデータ）を即座に表示
    selected_t = next((t for t in st.session_state['tables'] if isinstance(t, dict) and t.get('@id') == selected_table_id), None)
    if selected_t:
        gov_org_obj = selected_t.get('GOV_ORG', '')
        gov_org = gov_org_obj.get('$', '') if isinstance(gov_org_obj, dict) else str(gov_org_obj)
        
        cycle = selected_t.get('CYCLE', '設定なし')
        survey_date = str(selected_t.get('SURVEY_DATE', '不明'))
        if len(survey_date) > 4: survey_date = f"{survey_date[:4]}年{survey_date[4:]}月" # Basic formatting
        
        open_date = str(selected_t.get('OPEN_DATE', '不明'))
        if len(open_date) > 4: open_date = f"{open_date[:4]}年{open_date[4:6]}月{open_date[6:]}日"
        
        st.info(f"📋 **表の概要**\n- **作成機関**: {gov_org}\n- **調査年月**: {survey_date}\n- **調査周期**: {cycle}\n- **公開日**: {open_date}")
        
    
    if st.button("メタデータを取得し、AIと分析の相談を始める"):
        if not app_id_ready or not api_key_ready:
            st.error("e-Stat IDとGemini API Keyの両方が必要です。")
        else:
            with st.spinner("表の構造（メタデータ）を読み込んでいます..."):
                try:
                    meta_json = get_meta_info(selected_table_id, app_id=st.session_state['estat_app_id'])
                    class_objs = meta_json.get('GET_META_INFO', {}).get('METADATA_INF', {}).get('CLASS_INF', {}).get('CLASS_OBJ', [])
                    meta_summary = json.dumps(class_objs, ensure_ascii=False)
                    
                    st.session_state['meta_summary'] = meta_summary
                    st.session_state['selected_table_id_fixed'] = selected_table_id
                    st.session_state['chat_mode'] = True
                    
                    # Extract column names and examples for presentation
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
                        
                        if ex_str:
                            col_details.append(f"・**{col_name}**（例: {ex_str}）")
                        else:
                            col_details.append(f"・**{col_name}**")
                            
                    st.session_state['available_columns_details'] = col_details
                    
                    details_str = "\n".join(col_details)
                    st.session_state['messages'] = [
                        {"role": "assistant", "content": f"データ構造を読み込みました！このデータには以下の項目（カラム）が含まれています。\n\n{details_str}\n\nどの項目を、どのような条件（例：東京都のみ、など）で絞り込みたいかチャットで指示してください！"}
                    ]
                    st.session_state['filter_params'] = None
                except Exception as e:
                    st.error(f"メタデータ取得エラー: {e}")

st.divider()

if st.session_state.get('chat_mode'):
    st.subheader("💬 AIと絞り込みデータの相談")
    if 'available_columns_details' in st.session_state:
        details_md = "\n".join(st.session_state['available_columns_details'])
        st.info(f"🔰 **このデータで絞り込み可能な項目（と中身の例）:**\n\n{details_md}")
        
    for msg in st.session_state['messages']:
        st.chat_message(msg["role"]).write(msg["content"])
        
    prompt = st.chat_input("例: 条件を絞りたい（やり直す場合もここに入力して送信）")
    if prompt:
        # チャットが再送信されたら、現在のグラフと確定パラメータをクリア（やり直し状態にする）
        st.session_state['filter_params'] = None
        if 'current_df' in st.session_state:
            del st.session_state['current_df']
            
        st.session_state['messages'].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("AIが考え中..."):
                reply = chat_for_filtering(
                    messages=st.session_state['messages'],
                    meta_data_summary=st.session_state['meta_summary'],
                    api_key=st.session_state['gemini_api_key'],
                    model_name=llm_model
                )
                st.write(reply)
                st.session_state['messages'].append({"role": "assistant", "content": reply})
                
                extracted_params = extract_json_parameters(reply)
                if extracted_params:
                    st.session_state['filter_params'] = extracted_params
                    st.success(f"以下の条件でデータを取得します: \n{extracted_params}")
                st.rerun()

    if st.session_state.get('filter_params') is not None:
        if st.button("確定した条件でデータを取得してグラフ化！"):
            with st.spinner("e-Statから実データを取得・成型中..."):
                try:
                    raw_json = get_stats_data(
                        st.session_state['selected_table_id_fixed'], 
                        app_id=st.session_state['estat_app_id'], 
                        filter_params=st.session_state['filter_params']
                    )
                    df = parse_estat_json_to_dataframe(raw_json)
                    
                    if df is None or df.empty:
                        st.warning("この条件ではデータが0件でした。チャットを続けて別の条件を探してみてください。")
                    else:
                        st.session_state['current_df'] = df
                        st.success("データの取得に成功しました！")
                except Exception as e:
                    st.error(f"データ取得エラー: {e}")

if st.session_state.get('current_df') is not None:
    df_base = st.session_state['current_df']
    
    st.subheader("🧹 グラフ事前フィルタ（不要な総数や合計を除外）")
    st.markdown("ここで「総数」などのチェックを外すことで、意味のある美しいグラフが描かれます。")
    
    dimensions = [c for c in df_base.columns if c not in ['value', 'unit']]
    
    filters = {}
    if dimensions:
        cols = st.columns(min(len(dimensions), 4))
        for i, dim in enumerate(dimensions):
            col = cols[i % 4]
            unique_vals = [v for v in df_base[dim].dropna().unique().tolist() if v != '']
            
            # 「総数」や「計」を含む要素を初期状態から除外
            default_vals = [v for v in unique_vals if '総数' not in str(v) and '計' not in str(v) and '不詳' not in str(v)]
            if not default_vals:
                default_vals = unique_vals
                
            selected = col.multiselect(dim, unique_vals, default=default_vals)
            filters[dim] = selected
            
    # フィルタの適用
    df_filtered = df_base.copy()
    for dim, selected in filters.items():
        if selected != []:
            df_filtered = df_filtered[df_filtered[dim].isin(selected)]
        else:
            # 1つでも完全未選択の軸があればデータなしとする
            df_filtered = pd.DataFrame(columns=df_base.columns)

    st.subheader("📊 絞り込み後のデータプレビュー")
    st.dataframe(df_filtered.head())
    
    if 'value' in df_filtered.columns and not df_filtered.empty:
        st.subheader("📈 グラフ設定と描画")
        
        if dimensions:
            col1, col2, col3 = st.columns(3)
            chart_type = col1.selectbox("グラフの種類", ["折れ線グラフ", "棒グラフ", "円グラフ"])
            
            # 時間系の軸を自動推定
            time_keys = [d for d in dimensions if '時間' in d or '年' in d or '月' in d or '期' in d or 'time' in d.lower()]
            default_x = time_keys[0] if time_keys else dimensions[0]
            
            x_axis = col2.selectbox("X軸 / 円グラフ区分", dimensions, index=dimensions.index(default_x) if default_x in dimensions else 0)
            
            color_col_options = ["なし"] + [c for c in dimensions if c != x_axis]
            color_col = col3.selectbox("色分け（グループ）", color_col_options)
            
            df_plot = df_filtered.sort_values(by=x_axis)
            color_arg = color_col if color_col != "なし" else None
            
            try:
                if chart_type == "折れ線グラフ":
                    fig = px.line(df_plot, x=x_axis, y='value', color=color_arg, markers=True, title=f"{x_axis}別の推移")
                elif chart_type == "棒グラフ":
                    fig = px.bar(df_plot, x=x_axis, y='value', color=color_arg, title=f"{x_axis}別の比較", barmode='group')
                elif chart_type == "円グラフ":
                    fig = px.pie(df_plot, names=x_axis, values='value', title=f"{x_axis}別の割合")
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"現在選択中の条件ではグラフを描画できません。別の軸や条件を選んでみてください。（詳細: {e}）")
        else:
            st.info("グラフ化するための軸情報がデータに含まれていません。")
        
        st.subheader("🤖 AIによるインサイト（Gemini）")
        if st.button("このデータからインサイトを生成"):
            with st.spinner("AIが考察を生成中..."):
                summary = f"項目数: {len(df_filtered)}\n\n先頭データ:\n{df_filtered.head(10).to_string()}\n\n基本統計量:\n{df_filtered['value'].describe().to_string()}"
                insight = generate_insights(summary, api_key=st.session_state['gemini_api_key'], model_name=llm_model)
                st.markdown(f"> {insight}")
    else:
        st.info("データが空になる条件が選択されたか、valueカラムが存在しません。フィルタを見直してください。")
