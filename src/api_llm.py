import google.generativeai as genai
import json
import re

def generate_insights(data_summary: str, api_key: str, model_name: str = "gemini-1.5-flash"):
    """
    Generate insights using Google Gemini API. (Used after data is fetched)
    """
    if not api_key:
        return "⚠️ Gemini APIキーが設定されていません。"

    prompt = f"""
あなたはプロのデータアナリストです。
以下の日本の統計データの要約を見て、数値の変化のトレンドや、そこから推測できる社会的背景・インサイトを日本語でわかりやすく考察してください。

【データ要約】
{data_summary}
"""
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ インサイトの生成中にエラーが発生しました。\n\n詳細エラー: {str(e)}"

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
    return None
