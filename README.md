# 🇯🇵 e-Stat AI Analyzer

e-Stat（日本の政府統計の総合窓口）の膨大なデータを、**AI（Gemini）と会話しながら**直感的に絞り込み、自動で美しくグラフ化するエージェント型Webアプリケーションです。

🌐 **公開アプリURL (Streamlit Cloud)**  
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://estat-ai-analyzer.streamlit.app/)  
https://estat-ai-analyzer.streamlit.app/

---

## 💡 特徴 (Features)
- **AIアシスタント連携**: 何万もの項目がある巨大な政府統計でも、AIがメタデータ（構造）を読み取り、チャット形式で「どの軸を抽出するか」を提案・代行してくれます。不要なノイズ（「全国（総数）」の合算値など）によるグラフ崩壊を防ぎます。
- **2段階の絞り込み**: カテゴリ（例: 人口・世帯）とキーワード（例: 出生率）で表を特定し、その後AIとチャットで必要な内訳を柔軟に抽出します。
- **高度なセキュア通信**: ユーザーの e-Stat ID と Gemini API Key は、サーバーには保存されずブラウザの `LocalStorage` にのみ安全に保存・保持されます。
- **動的グラフ化エージェント**: 取得したデータを「折れ線」「棒グラフ」「円グラフ」に自動描画。X軸や色分け（グループ）をワンタッチで切り替えられます。

## 🛠️ 技術スタック
- **Python 3.9+**
- **Streamlit**: WebUIとインタラクション
- **Google Generative AI (Gemini)**: パラメータ推論とデータ分析インサイトの生成
- **Plotly**: インタラクティブなグラフ描画
- **Pandas**: 複雑なe-Stat JSONのテーブル成型

## 🚀 ローカルでの動かし方 (Installation)

1. リポジトリのクローン
   ```bash
   git clone https://github.com/takedayasushi/estat-ai-analyzer.git
   cd estat-ai-analyzer
   ```

2. 依存関係のインストール
   ```bash
   pip install -r requirements.txt
   ```

3. アプリの起動
   ```bash
   streamlit run app.py
   ```

## 🔒 セキュリティについて
本アプリ（Streamlit Community Cloud上）にて入力されたAPIキー情報は、ブラウザ側（LocalStorage）のみで保持され、開発者やサーバー側がログを収集することはありません。ご自身のキーで安全に分析機能をご利用いただけます。
