import sys
# Streamlit Cloudの強力すぎるモジュールキャッシュを強制削除するハック（ImportError対策）
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

# --- Version Info (For Debugging) ---
APP_VERSION = "2026-04-19-0905"

# --- basic configurations ---
st.set_page_config(page_title=f"e-Stat AI Analyzer v{APP_VERSION}", layout="wide")
localS = LocalStorage()

# --- State Initialization (Critical: Prevent "missing" errors) ---
INIT_KEYS = {
    'estat_app_id': '',
    'gemini_api_key': '',
    'messages': [],
    'insight_messages': [],
    'current_df': None,
    'chat_mode': False,
    'filter_params': None,
    'selected_table_id_fixed': None,
    'meta_summary': '',
    'available_columns_details': [],
    'chart_config': {},
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

# --- core logic functions ---
def setup_analysis_phase(selected_table_id):
    if not st.session_state.get('estat_app_id') or not st.session_state.get('gemini_api_key'):
        st.error("API設定が不足しています。")
        return False
    with st.spinner("メタデータを読み込み中..."):
        try:
            meta_json = get_meta_info(selected_table_id, app_id=st.session_state['estat_app_id'])
            raw = meta_json.get('GET_META_INFO', {}).get('METADATA_INF', {}).get('CLASS_INF', {})
            class_objs = raw.get('CLASS_OBJ', [])
            st.session_state['meta_summary'] = json.dumps(class_objs, ensure_ascii=False)
            st.session_state['selected_table_id_fixed'] = selected_table_id
            st.session_state['chat_mode'] = True
            st.session_state['active_analysis_id'] = str(uuid.uuid4())
            st.session_state['insight_messages'] = []
            return True
        except Exception as e:
            st.error(f"エラー: {e}")
            return False

def restore_saved_analysis(item):
    app_id = st.session_state.get('estat_app_id')
    if not app_id: 
        st.error("e-Stat IDを設定してください。")
        return False
        
    with st.spinner("🔄 データを読み込み中..."):
        try:
            st.session_state['selected_table_id_fixed'] = item['table_id']
            st.session_state['filter_params'] = item['filter_params']
            st.session_state['chart_config'] = item.get('chart_config', {})
            st.session_state['dimension_filters'] = item.get('dimension_filters', {})
            
            raw_json = get_stats_data(item['table_id'], app_id=app_id, filter_params=item['filter_params'])
            df = parse_estat_json_to_dataframe(raw_json)
            if df is not None:
                st.session_state['current_df'] = df
            
            st.session_state['active_analysis_id'] = str(uuid.uuid4())
            st.session_state['chat_mode'] = True
            
            for k in ['manual_tables', 'ai_recommendations', 'ai_search_query']:
                if k in st.session_state: del st.session_state[k]

            st.toast("✅ 復元成功")
            return True
        except Exception as e:
            st.error(f"復元エラー: {e}")
            return False

# --- UI Setup ---
st.title(f"📊 e-Stat AI Analyzer (v{APP_VERSION})")

# --- Settings Sidebar ---
with st.sidebar.expander("⚙️ API設定"):
    try:
        s_e = localS.getItem("estat_app_id")
        s_g = localS.getItem("gemini_api_key")
    except: s_e = s_g = None
    if s_e and not st.session_state['estat_app_id']: st.session_state['estat_app_id'] = s_e
    if s_g and not st.session_state['gemini_api_key']: st.session_state['gemini_api_key'] = s_g
    c_e = st.text_input("e-Stat ID", value=st.session_state['estat_app_id'], type="password")
    c_g = st.text_input("Gemini API Key", value=st.session_state['gemini_api_key'], type="password")
    if st.button("保存"):
        localS.setItem("estat_app_id", c_e); localS.setItem("gemini_api_key", c_g)
        st.session_state['estat_app_id'] = c_e; st.session_state['gemini_api_key'] = c_g
        st.rerun()

models = fetch_gemini_models(st.session_state['gemini_api_key'])
llm_model = st.sidebar.selectbox("AIモデル", models if models else ["gemini-1.5-flash"])

# --- Gallery Sidebar ---
st.sidebar.divider()
with st.sidebar.expander("🔖 ブックマーク/共有"):
    if st.button("🚨 全体削除"): global_gallery.clear(); st.rerun()
    for i, item in enumerate(reversed(global_gallery)):
        if st.button(f"🔗 {item.get('title')}", key=f"g_{i}"):
            if restore_saved_analysis(item): st.rerun()
    st.markdown("---")
    try:
        bs = json.loads(localS.getItem("estat_my_bookmarks") or "[]")
    except: bs = []
    for i, item in enumerate(reversed(bs)):
        if st.button(f"📄 {item.get('title','無題')}", key=f"l_{i}"):
            if restore_saved_analysis(item): st.rerun()

# --- Search Tabs ---
st.divider()
t1, t2 = st.tabs(["🤖 AI検索", "📚 手動検索"])
with t2:
    sc = st.selectbox("カテゴリ", list(ESTAT_CATEGORIES.keys()))
    kw = st.text_input("キーワード")
    if st.button("統計表を検索", key="manual_search_btn"):
        res = search_stats_list(ESTAT_CATEGORIES[sc], st.session_state['estat_app_id'], kw)
        if res: st.session_state['manual_tables'] = res
    if 'manual_tables' in st.session_state:
        opts = {f"{t.get('TITLE',{}).get('$', t.get('TITLE',''))}": t.get('@id') for t in st.session_state['manual_tables']}
        sn = st.selectbox("結果を選択", list(opts.keys()))
        if st.button("この表で解析開始", key="manual_start_btn"):
            if setup_analysis_phase(opts[sn]): st.rerun()

with t1:
    if st.session_state['chat_mode']: st.info("分析実行中")
    else:
        aq = st.chat_input("調べたいことを入力")
        if aq:
            st.session_state['ai_search_query'] = aq
            with st.spinner("AIが探索中..."):
                p = generate_search_query(aq, ", ".join(ESTAT_CATEGORIES.keys()), st.session_state['gemini_api_key'], llm_model)
                if p:
                    raw = search_stats_list(p['category_id'], st.session_state['estat_app_id'], p.get('search_keyword',''))
                    recs = recommend_tables_from_list(aq, raw, st.session_state['gemini_api_key'], llm_model)
                    if recs: st.session_state['ai_recommendations'] = recs
        if 'ai_recommendations' in st.session_state:
            for rec in st.session_state['ai_recommendations']:
                with st.container(border=True):
                    st.write(f"🏅 {rec.get('title')}")
                    if st.button("この表で開始", key=f"ais_btn_{rec.get('id')}"):
                        if setup_analysis_phase(rec.get('id')): st.rerun()

# --- Analysis Phase ---
if st.session_state['chat_mode']:
    if st.session_state['active_analysis_id'] != st.session_state['last_processed_id']:
        st.session_state['last_processed_id'] = st.session_state['active_analysis_id']
        now = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state['insight_messages'] = [{
            "role": "user", 
            "content": f"【解析リクエスト: {now}】このデータの傾向とインサイトを分析してください。"
        }]
        st.rerun()

    st.divider()
    st.subheader("💬 データの絞り込み")
    for msg in st.session_state['messages']: st.chat_message(msg["role"]).write(msg["content"])
    p = st.chat_input("絞り込み条件を入力（例：東京都）")
    if p:
        st.session_state['messages'].append({"role": "user", "content": p})
        with st.chat_message("assistant"):
            rep = chat_for_filtering(st.session_state['messages'], st.session_state['meta_summary'], st.session_state['gemini_api_key'], llm_model)
            st.session_state['messages'].append({"role": "assistant", "content": rep})
            ext = extract_json_parameters(rep)
            if ext: st.session_state['filter_params'] = ext
            st.rerun()

    if st.session_state['filter_params'] and st.button("グラフを表示/更新"):
        raw = get_stats_data(st.session_state['selected_table_id_fixed'], st.session_state['estat_app_id'], st.session_state['filter_params'])
        df = parse_estat_json_to_dataframe(raw)
        if df is not None: st.session_state['current_df'] = df; st.rerun()

if st.session_state['current_df'] is not None:
    df_b = st.session_state['current_df']
    st.subheader("📊 データの確認と可視化")
    dims = [c for c in df_b.columns if c not in ['value', 'unit']]
    filters = {}
    if dims:
        cols = st.columns(min(len(dims), 4))
        for i, d in enumerate(dims):
            uvs = [v for v in df_b[d].dropna().unique() if v != '']
            s_f = st.session_state['dimension_filters'].get(d)
            defs = [v for v in uvs if v in s_f] if s_f else [v for v in uvs if not any(k in str(v) for k in ['総数','計'])]
            filters[d] = cols[i%4].multiselect(d, uvs, default=defs or uvs)
    
    df_f = df_b.copy()
    for d, v in filters.items():
        if v: df_f = df_f[df_f[d].isin(v)]
    if df_f.empty and not df_b.empty: df_f = df_b.head(50)

    # Chart
    c_opts = ["折れ線", "棒", "円"]
    sc = st.session_state['chart_config'].get('chart_type')
    ct = st.selectbox("グラフの種類", c_opts, index=c_opts.index(sc) if sc in c_opts else 0)
    
    if not df_f.empty:
        st.dataframe(df_f.head())
        fig = px.line(df_f, x=dims[0], y='value')
        if ct == "棒": fig = px.bar(df_f, x=dims[0], y='value')
        elif ct == "円": fig = px.pie(df_f, names=dims[0], values='value')
        st.plotly_chart(fig, use_container_width=True)

    # --- AI Analysis ---
    st.divider()
    st.subheader("💡 AIアナリストの解析レポート")
    
    # 手動更新用の小さなボタン
    if st.button("🔄 AIに再解析を依頼"):
        st.session_state['last_processed_id'] = "" # Reset to force refresh
        st.rerun()

    for m in st.session_state['insight_messages']:
        st.chat_message(m["role"]).write(m["content"])
    
    if st.session_state['insight_messages'] and st.session_state['insight_messages'][-1]["role"] == "user":
        with st.spinner("AIが最新データを読み取り中..."):
            summary = f"データ数: {len(df_f)}\n項目:\n{df_f.head(5).to_string()}"
            res = chat_for_insights(st.session_state['insight_messages'], summary, st.session_state['gemini_api_key'], llm_model)
            st.session_state['insight_messages'].append({"role": "assistant", "content": res})
            st.rerun()

    # Save Bookmark (Recipes only)
    st.divider()
    if st.button("💾 この可視化設定を保存"):
        item = {
            "title": f"保存: {datetime.datetime.now().strftime('%H:%M')}",
            "table_id": st.session_state['selected_table_id_fixed'],
            "filter_params": st.session_state['filter_params'],
            "insight_messages": [], 
            "chart_config": {"chart_type": ct},
            "dimension_filters": filters,
            "timestamp": datetime.datetime.now().strftime("%m/%d %H:%M")
        }
        l = json.loads(localS.getItem("estat_my_bookmarks") or "[]")
        l.append(item)
        localS.setItem("estat_my_bookmarks", json.dumps(l, ensure_ascii=False))
        st.success("ブックマークに保存しました。")
