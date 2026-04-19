import google.generativeai as genai
import json
import re

def chat_for_insights(messages: list, data_summary: str, api_key: str, model_name: str = "gemini-1.5-flash"):
    """
    Interactive Q&A and insight generation using Google Gemini API.
    """
    if not api_key:
        return "⚠️ Gemini APIキーが設定されていません。"

    system_instruction = f"""
あなたはプロのデータアナリストです。
以下の日本の統計データの要約（実データ）を踏まえて、ユーザーからの質問に答えたり、数値の変化のトレンド、推測できる社会的背景・インサイトを日本語でわかりやすく考察してください。
ユーザーが「考察して」と入力した場合は、全体を俯瞰したプロのレポートを出力してください。詳細な質問が来た場合は、データに基づいて正確に答えてください。

【現在グラフ化されている実データの要約情報】
{data_summary}
"""
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        
        gemini_history = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [m["content"]]})
            
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(messages[-1]["content"])
        return response.text
    except Exception as e:
        return f"⚠️ 考察の生成中にエラーが発生しました。\n\n詳細エラー: {str(e)}"

def chat_for_filtering(messages: list, meta_data_summary: str, api_key: str, model_name: str = "gemini-1.5-flash"):
    """
    Chat with the user to narrow down e-Stat filter parameters before fetching large datasets.
    """
    if not api_key:
        return "⚠️ Gemini APIキーが設定されていません。"

    system_instruction = f"""
あなたは日本の政府統計（e-Stat）のデータを絞り込むためのデータアナリストアシスタントです。
統計データは全件取得すると巨大すぎるため、対象のメタデータをもとに、ユーザーの目的（どの地域、年代、切り口を見たいか）を聞き出し、API取得用のパラメータを決定するのがあなたの仕事です。

【対象の統計表 メタデータ】
{meta_data_summary}

【ステップ】
1. ユーザーのざっくりした要望に対して、「このデータには〇〇や〇〇の軸があります。地域は東京都だけに絞りますか？」など提案しながら対話します。
2. e-Statの絞り込みパラメータは、通常 `cdArea` (地域), `cdTime` (時間), `cdCat01`等 (カテゴリ) といったIDのキーに、対応する `@code` の値を指定します。
3. ユーザーの要件が確定したら、最後に必ず以下のフォーマットでAPI用パラメータのJSONを出力して会話を終えてください。
```json
{{
  "cdArea": "13000",
  "cdTime": "2020"
}}
```
"""

    try:
        genai.configure(api_key=api_key)
        # Create model with system instructions
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        
        # Convert Streamlit message format to Gemini format
        gemini_history = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [m["content"]]})
            
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(messages[-1]["content"])
        return response.text
    except Exception as e:
        return f"⚠️ エラーが発生しました: {str(e)}"

def extract_json_parameters(text: str):
    """
    Helper to extract the JSON block from Gemini's response containing API parameters.
    """
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 括弧だけのJSONブロックがない場合もフォールバックでパースを試みる
    try:
        # {} や [] の抽出
        match_obj = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match_obj:
            return json.loads(match_obj.group(1))
    except:
        pass
    return None

def generate_search_query(user_intent: str, categories_str: str, api_key: str, model_name: str = "gemini-1.5-flash"):
    """
    Translates natural language user intent into an e-Stat API search query (category and keyword).
    """
    if not api_key:
        return None
    
    prompt = f"""
あなたは日本の政府統計（e-Stat）の専門データアナリストアシスタントです。
ユーザーの「知りたいこと（自然言語）」を分析し、e-StatのAPIを使って検索するための「大分類カテゴリ番号」と「検索キーワード」を推論してください。

【ユーザーの知りたいこと】
{user_intent}

【選択可能な大分類カテゴリ】
{categories_str}

【e-Stat検索における極めて重要な注意点】
e-Statの検索APIは「統計表タイトル・概要」との単純な文字列マッチであるため、自然言語の文脈を含めると容赦なく「該当なし(0件)」になります。
絶対に以下のタイプのワードをキーワードに含めないでください。
- 「推移」「変化」「前後」「比較」「割合」「ランキング」などの分析・状態ワード
- 「コロナ」「最近」「近年」などの時代・事象ワード
- 「知りたい」「データ」「高齢者」などの口語や過度な具体化ワード（統計表名はもっと抽象的です）
抽出するのは **「死亡」「人口」「労働」「家計」といった、お堅い統計の基本名詞（1〜2語）のみ** としてください。絞り込みすぎると失敗するため、迷ったらカテゴリ指定のみ（キーワードは空文字 `""`）が最強の戦略です。

【出力ルール】
必ず以下のフォーマットのJSONで出力してください。
- category_id: 最も適切な大分類の2桁の数字（例: "02"）
- search_keyword: 上記の注意点に従った、極めてシンプルで抽象的な基本名詞（例: "死亡"）。最大でも2語まで（スペース区切り）。

出力例（例外的な入力「コロナ前後の高齢者の死亡者数推移」という質問への正解）:
```json
{{
    "category_id": "02",
    "search_keyword": "死亡"
}}
```
"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return extract_json_parameters(response.text)
    except Exception as e:
        print(e)
        return None

def recommend_tables_from_list(user_intent: str, tables: list, api_key: str, model_name: str = "gemini-1.5-flash"):
    """
    Takes a massive list of table dictionaries from e-Stat and uses Gemini to pick the top 3-5 best tables.
    """
    if not api_key or not tables:
        return []
    
    # Token節約のため、最大50件に制限し、必要な情報だけ抽出する
    simplified_tables = []
    for t in tables[:50]:
        simplified_tables.append({
            "id": t.get('@id', ''),
            "stat_name": t.get('STAT_NAME', {}).get('$', '') if isinstance(t.get('STAT_NAME'), dict) else t.get('STAT_NAME', ''),
            "table_name": t.get('TABLE_NAME', {}).get('$', '') if isinstance(t.get('TABLE_NAME'), dict) else t.get('TABLE_NAME', ''),
            "survey_date": t.get('SURVEY_DATE', '')
        })
        
    tables_json_str = json.dumps(simplified_tables, ensure_ascii=False)

    prompt = f"""
あなたは日本の政府統計（e-Stat）の専門データコンシェルジュです。
ユーザーの目的に対してe-Statで検索をかけたところ、以下の統計表リストがヒットしました。
この中から、ユーザーの目的に最も合致しそうな素晴らしい統計表を **最大5つ**（1つの場合も可）厳選し、その推薦理由とともにJSONで返してください。

【ユーザーの目的】
{user_intent}

【ヒットした統計表リスト（一部抽出）】
{tables_json_str}

【出力ルール】
必ず以下のフォーマットのJSONのリスト（配列）を出力してください。
```json
[
  {{
    "id": "抽出した表のid",
    "title": "統計表の正式タイトル",
    "reason": "なぜこの表が目的に一番合致しているか、ユーザーへの推薦コメント（日本語で簡潔に）"
  }}
]
```
"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return extract_json_parameters(response.text)
    except Exception as e:
        print(f"Error in recommend_tables: {e}")
        return []
