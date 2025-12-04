"""
八門遁甲マーケットラジオ音声生成スクリプト（8-mon-assets版）
VOICEVOX (ずんだもん + 四国めたん) を使用して市場解説音声を生成

音声: VOICEVOX:ずんだもん、VOICEVOX:四国めたん
- VOICEVOX: https://voicevox.hiroshiba.jp/
- キャラクター利用規約: https://zunko.jp/con_ongen_kiyaku.html
"""
import os
import io
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import argparse
from dataclasses import dataclass

import requests
from pydub import AudioSegment
import numpy as np
import urllib.request
import re
from bs4 import BeautifulSoup

# パス設定
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# VOICEVOX設定
VOICEVOX_URL = os.environ.get("VOICEVOX_URL", "http://localhost:50021")
SPEAKERS = {
    "metan": 2,      # 四国めたん（ノーマル）
    "zundamon": 3,   # ずんだもん（ノーマル）
}


@dataclass
class DialogueLine:
    """対話の1行を表す"""
    speaker: str  # "metan" or "zundamon"
    text: str


def synthesize_voicevox(text: str, speaker_id: int) -> bytes:
    """VOICEVOXで音声合成"""
    # 1. audio_query
    query_resp = requests.post(
        f"{VOICEVOX_URL}/audio_query",
        params={"text": text, "speaker": speaker_id},
        timeout=30
    )
    query_resp.raise_for_status()
    query = query_resp.json()

    # 2. synthesis
    synth_resp = requests.post(
        f"{VOICEVOX_URL}/synthesis",
        params={"speaker": speaker_id},
        json=query,
        timeout=60
    )
    synth_resp.raise_for_status()
    return synth_resp.content  # WAV bytes


def get_jst_time() -> str:
    """JST現在時刻を取得"""
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).strftime("%H時%M分")


def get_jst_date() -> str:
    """JST現在日付を取得（読み上げ用）"""
    jst = timezone(timedelta(hours=9))
    now = datetime.now(jst)
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    return f"{now.month}月{now.day}日{weekdays[now.weekday()]}曜日"


def fetch_weather() -> dict:
    """気象庁APIから東日本・西日本の天気を取得"""
    # 東京（130000）と大阪（270000）の天気を代表として使用
    weather_info = {"east": "", "west": ""}

    try:
        # 東京の天気
        req = urllib.request.Request(
            "https://www.jma.go.jp/bosai/forecast/data/forecast/130000.json",
            headers={"User-Agent": "GaryuRadio/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # 今日の天気を取得
            weather = data[0]["timeSeries"][0]["areas"][0]["weathers"][0]
            # 長い天気説明を短縮
            weather_short = weather.replace("　", "").replace("所により", "一部")[:15]
            weather_info["east"] = weather_short
            print(f"  [Weather] East (Tokyo): {weather_short}")
    except Exception as e:
        print(f"  Warning: Failed to fetch East weather: {e}")
        weather_info["east"] = "情報取得中"

    try:
        # 大阪の天気
        req = urllib.request.Request(
            "https://www.jma.go.jp/bosai/forecast/data/forecast/270000.json",
            headers={"User-Agent": "GaryuRadio/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            weather = data[0]["timeSeries"][0]["areas"][0]["weathers"][0]
            weather_short = weather.replace("　", "").replace("所により", "一部")[:15]
            weather_info["west"] = weather_short
            print(f"  [Weather] West (Osaka): {weather_short}")
    except Exception as e:
        print(f"  Warning: Failed to fetch West weather: {e}")
        weather_info["west"] = "情報取得中"

    return weather_info


def format_price_for_tts(price: float) -> str:
    """価格を読み上げやすい形式に変換（ゴマン問題対策）

    50000 → "5万"
    49864 → "4万9864"
    """
    p = int(price)
    if p >= 10000:
        man = p // 10000
        remainder = p % 10000
        if remainder == 0:
            return f"{man}万"
        return f"{man}万{remainder}"
    return str(p)


def normalize_for_tts(text: str) -> str:
    """TTS用にテキストを正規化（読み替え）"""
    replacements = {
        "BTC": "ビットコイン",
        "日経225": "ニッケイニーニーゴ",
        "CALL": "コール",
        "PUT": "プット",
        "USD/JPY": "ドル円",
        "NASDAQ": "ナスダック",
        "GOLD": "ゴールド",
        "臥龍": "がりょう",
    }
    for original, reading in replacements.items():
        text = text.replace(original, reading)
    return text


def get_api_headers() -> dict:
    """API呼び出し用のヘッダーを取得（API Key認証対応）"""
    headers = {"User-Agent": "GaryuRadio/1.0"}
    api_key = os.environ.get("GARYU_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def fetch_market_live() -> dict:
    """本番APIから市場データを取得"""
    try:
        req = urllib.request.Request(
            "https://8-mon.com/api/market-live",
            headers={"User-Agent": "GaryuRadio/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"  [API] Market: {data.get('marketSession')} @ {data.get('price')}")
            return data
    except Exception as e:
        print(f"  Warning: Failed to fetch market data: {e}")
        return {}


def fetch_news() -> list[dict]:
    """本番APIからニュースを取得"""
    try:
        req = urllib.request.Request(
            "https://8-mon.com/api/garyu-news",
            headers=get_api_headers()
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # 重要度3以上の上位3件
            news = data.get("news", [])
            filtered = [n for n in news if n.get("importance", 0) >= 3][:3]
            print(f"  [API] News: {len(filtered)} items (importance >= 3)")
            return filtered
    except Exception as e:
        print(f"  Warning: Failed to fetch news: {e}")
        return []


def fetch_ai_comment(section: str) -> str | None:
    """本番APIからAIコメントを取得"""
    try:
        if section == "option":
            url = "https://8-mon.com/api/garyu-option-oi"
        else:
            url = f"https://8-mon.com/api/garyu-comment?section={section}"

        req = urllib.request.Request(url, headers=get_api_headers())
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            comment = data.get("comment", "")
            if comment and data.get("source") != "fallback":
                print(f"  [API] AI comment for {section}: loaded")
                return comment
    except Exception as e:
        print(f"  Warning: Failed to fetch AI comment for {section}: {e}")
    return None


def fetch_integrated_metrics() -> dict:
    """本番APIから統合メトリクスを取得"""
    try:
        req = urllib.request.Request(
            "https://8-mon.com/integrated-metrics.json",
            headers={"User-Agent": "GaryuRadio/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  Warning: Failed to fetch integrated metrics: {e}")
        return {}


def fetch_data_json() -> list:
    """本番APIから市場データを取得"""
    try:
        req = urllib.request.Request(
            "https://8-mon.com/data.json",
            headers={"User-Agent": "GaryuRadio/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  Warning: Failed to fetch data.json: {e}")
        return []


def fetch_correlation_matrix() -> dict:
    """本番APIから相関マトリクスを取得"""
    try:
        req = urllib.request.Request(
            "https://8-mon.com/correlation-matrix.json",
            headers={"User-Agent": "GaryuRadio/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  Warning: Failed to fetch correlation matrix: {e}")
        return {}


def fetch_option_oi() -> dict:
    """本番APIからオプション建玉を取得"""
    try:
        req = urllib.request.Request(
            "https://8-mon.com/option-oi.json",
            headers={"User-Agent": "GaryuRadio/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  Warning: Failed to fetch option OI: {e}")
        return {}


def scrape_google_news_nikkei() -> list[str]:
    """Google Newsから日経関連ニュースをスクレイピング"""
    try:
        # URLエンコード済みクエリ（日経平均 OR 日経225 OR 株価）
        url = "https://news.google.com/rss/search?q=%E6%97%A5%E7%B5%8C%E5%B9%B3%E5%9D%87+OR+%E6%97%A5%E7%B5%8C225+OR+%E6%A0%AA%E4%BE%A1&hl=ja&gl=JP&ceid=JP:ja"
        req = urllib.request.Request(url, headers={"User-Agent": "GaryuRadio/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8")

        # RSS XMLをパース（lxml-xmlが使えない場合は正規表現にフォールバック）
        try:
            soup = BeautifulSoup(content, "lxml-xml")
            items = soup.find_all("item")[:10]
            headlines = []
            for item in items:
                title = item.find("title")
                if title:
                    headlines.append(title.text.strip())
            if headlines:
                print(f"  [Scrape] Fetched {len(headlines)} headlines from Google News (lxml)")
                return headlines
        except Exception as e:
            print(f"  [Scrape] lxml-xml failed: {e}, using regex fallback")

        # フォールバック: 正規表現でタイトル抽出（CDATA対応）
        titles = re.findall(r'<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>', content, re.DOTALL)
        # 最初の2件をスキップ（フィードタイトル + "Google ニュース"）
        headlines = []
        for t in titles[2:]:  # 最初の2件をスキップ
            cleaned = t.strip().replace('\n', ' ')
            if cleaned and len(headlines) < 10:
                headlines.append(cleaned)
        print(f"  [Scrape] Fetched {len(headlines)} headlines (regex fallback)")
        return headlines
    except Exception as e:
        print(f"  Warning: Failed to scrape Google News: {e}")
        return []


def fetch_market_analysis(headlines: list[str], price_change: float, price_change_pct: float) -> dict:
    """Workers AI APIで市場要因分析を取得"""
    try:
        payload = json.dumps({
            "headlines": headlines,
            "priceChange": price_change,
            "priceChangePercent": price_change_pct,
        }).encode("utf-8")

        headers = {
            "User-Agent": "GaryuRadio/1.0",
            "Content-Type": "application/json",
        }
        # API Key認証（環境変数から取得）
        api_key = os.environ.get("GARYU_API_KEY")
        if api_key:
            headers["X-API-Key"] = api_key

        req = urllib.request.Request(
            "https://8-mon.com/api/garyu-market-analysis",
            data=payload,
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"  [API] Market analysis: {data.get('source')}")
            return data
    except Exception as e:
        print(f"  Warning: Failed to fetch market analysis: {e}")
        return {"bullishFactors": [], "bearishFactors": [], "summary": ""}


# ============================================================
# 対話台本生成関数
# ============================================================

def generate_opening(weather: dict = None) -> list[DialogueLine]:
    """オープニング（日付・天気付き）"""
    time = get_jst_time()
    date = get_jst_date()

    # 天気情報がない場合は取得
    if weather is None:
        weather = fetch_weather()

    east_weather = weather.get("east", "")
    west_weather = weather.get("west", "")

    lines = [
        DialogueLine("metan", f"皆さんこんにちは。八門遁甲マーケットラジオの時間です。"),
        DialogueLine("zundamon", f"ずんだもんなのだ！今日も一緒に市場を見ていくのだ！"),
        DialogueLine("metan", f"本日は{date}、現在時刻は{time}です。"),
    ]

    # 天気情報を追加
    if east_weather and west_weather:
        lines.extend([
            DialogueLine("zundamon", f"今日のお天気はどうなのだ？"),
            DialogueLine("metan", f"東日本は{east_weather}、西日本は{west_weather}となっています。"),
        ])
    elif east_weather:
        lines.append(DialogueLine("metan", f"今日の東日本は{east_weather}です。"))
    elif west_weather:
        lines.append(DialogueLine("metan", f"今日の西日本は{west_weather}です。"))

    lines.extend([
        DialogueLine("zundamon", f"なるほどなのだ！で、市場はどうなってるのだ？"),
        DialogueLine("metan", f"では早速、本日の市況から見ていきましょう。"),
    ])

    return lines


def generate_market_summary(market_data: dict = None, news: list[dict] = None) -> list[DialogueLine]:
    """市況サマリー"""
    # APIデータがある場合はそれを使用
    if market_data and market_data.get("price"):
        price = market_data.get("price", 0)
        session = market_data.get("marketSession", "closed")
        change = market_data.get("change", 0)
        change_pct = market_data.get("changePercent", 0)
    else:
        # フォールバック: data.jsonから取得
        data = fetch_data_json()
        if not data:
            return [
                DialogueLine("metan", "市場データを取得中です。"),
                DialogueLine("zundamon", "待つのだ！"),
            ]
        latest = data[-1]
        price = latest.get("Nikkei_Close", 0)
        session = "closed"
        if len(data) >= 2:
            prev = data[-2].get("Nikkei_Close", price)
            change = price - prev
            change_pct = (change / prev) * 100 if prev else 0
        else:
            change = 0
            change_pct = 0

    price_str = format_price_for_tts(price)
    market_type = "日経先物" if session == "futures" else "日経平均"

    lines = []

    if change > 0:
        change_str = format_price_for_tts(abs(change))
        lines = [
            DialogueLine("zundamon", f"今日の{market_type}はどうなっているのだ？"),
            DialogueLine("metan", f"{market_type}は現在{price_str}円です。"),
            DialogueLine("zundamon", f"おお！上がってるのだ？"),
            DialogueLine("metan", f"はい、前日比プラス{change_str}円、プラス{abs(change_pct):.2f}パーセントで推移しています。"),
            DialogueLine("zundamon", f"やったのだ！調子いいのだ！"),
            DialogueLine("metan", f"ただし、上昇相場でも油断は禁物ですよ。"),
        ]
    elif change < 0:
        change_str = format_price_for_tts(abs(change))
        lines = [
            DialogueLine("zundamon", f"今日の{market_type}はどうなっているのだ？"),
            DialogueLine("metan", f"{market_type}は現在{price_str}円です。"),
            DialogueLine("zundamon", f"えっ、下がってるのだ？"),
            DialogueLine("metan", f"はい、前日比マイナス{change_str}円、マイナス{abs(change_pct):.2f}パーセントです。"),
            DialogueLine("zundamon", f"ちょっと心配なのだ。大丈夫なのだ？"),
            DialogueLine("metan", f"下落局面はチャンスでもあります。冷静に見極めましょう。"),
        ]
    else:
        lines = [
            DialogueLine("zundamon", f"今日の{market_type}はどうなっているのだ？"),
            DialogueLine("metan", f"{market_type}は現在{price_str}円。ほぼ横ばいですね。"),
            DialogueLine("zundamon", f"動きがないのだ？"),
            DialogueLine("metan", f"静かな相場の後には大きな動きが来ることもあります。注視しましょう。"),
        ]

    # ニュース追加
    if news:
        lines.append(DialogueLine("metan", "続いて、本日の注目ニュースです。"))
        lines.append(DialogueLine("zundamon", "気になるのだ！教えてほしいのだ！"))
        for i, item in enumerate(news, 1):
            headline = item.get("headline", "")
            lines.append(DialogueLine("metan", f"{i}件目、{headline}"))
            if i == 1:
                lines.append(DialogueLine("zundamon", "それは重要なのだ？"))
                lines.append(DialogueLine("metan", "市場に影響を与える可能性がありますね。"))

    return lines


def generate_market_analysis(market_data: dict) -> list[DialogueLine]:
    """市場要因分析コーナー（Workers AIで上昇/下降要因を分析）"""
    price_change = market_data.get("change", 0)
    price_change_pct = market_data.get("changePercent", 0)

    # Google Newsからニュースを取得
    headlines = scrape_google_news_nikkei()
    if not headlines:
        return [
            DialogueLine("metan", "本日の要因分析ですが、ニュースの取得に時間がかかっています。"),
            DialogueLine("zundamon", "次のコーナーに進むのだ！"),
        ]

    # Workers AIで分析
    analysis = fetch_market_analysis(headlines, price_change, price_change_pct)

    bullish = analysis.get("bullishFactors", [])
    bearish = analysis.get("bearishFactors", [])
    summary = analysis.get("summary", "")

    lines = [
        DialogueLine("metan", "続いて、本日の相場要因を分析していきましょう。"),
        DialogueLine("zundamon", "なんで上がったり下がったりするのか知りたいのだ！"),
    ]

    # 上昇要因
    if bullish:
        lines.append(DialogueLine("metan", "まず上昇要因から。"))
        for i, factor in enumerate(bullish[:3], 1):
            lines.append(DialogueLine("metan", f"{i}つ目、{factor}"))
        lines.append(DialogueLine("zundamon", "なるほど、それで上がってるのだ！"))

    # 下降要因
    if bearish:
        lines.append(DialogueLine("metan", "一方、下落圧力となっている要因もあります。"))
        for i, factor in enumerate(bearish[:3], 1):
            lines.append(DialogueLine("metan", f"{i}つ目、{factor}"))
        lines.append(DialogueLine("zundamon", "そっちも気になるのだ。"))

    # 総括
    if summary:
        lines.append(DialogueLine("metan", f"総合すると、{summary}"))
        lines.append(DialogueLine("zundamon", "よく分かったのだ！ありがとうなのだ！"))

    if not bullish and not bearish:
        lines.extend([
            DialogueLine("metan", "本日は明確な材料に乏しい相場ですね。"),
            DialogueLine("zundamon", "様子見なのだ？"),
            DialogueLine("metan", "次の材料を待つ局面かもしれません。"),
        ])

    return lines


def generate_sentiment() -> list[DialogueLine]:
    """センチメント分析（信用倍率ベース）

    データソース: 日経レバETF（1570）の信用倍率
    - 信用買い残 ÷ 信用売り残 = 信用倍率
    - 週次で東証が公表するデータを使用
    """
    # データ取得
    metrics = fetch_integrated_metrics()
    cr_data = metrics.get("credit_ratio", {}).get("latest", {})
    cr = cr_data.get("value", 1.2)
    percentile = cr_data.get("percentile", 50)

    # AIコメント取得
    cached = fetch_ai_comment("feargreed")

    # === 導入パート: データソースの説明 ===
    lines = [
        DialogueLine("metan", "次は市場センチメント、フィアーアンドグリード指数を見ていきましょう。"),
        DialogueLine("zundamon", "フィアーアンドグリード？恐怖と強欲なのだ？"),
        DialogueLine("metan", "はい。このサイトでは日経レバETF、銘柄コード1570の信用倍率をもとに算出しています。"),
        DialogueLine("zundamon", "なんで日経レバなのだ？"),
        DialogueLine("metan", "日経レバは日経平均の2倍の値動きをするETFで、個人投資家に大人気なんです。"),
        DialogueLine("zundamon", "みんな買ってるのだ！"),
        DialogueLine("metan", "そうなんです。だからこのETFの信用取引の状況を見れば、個人投資家の心理がわかるんですね。"),
    ]

    # === 統計解説パート ===
    lines.extend([
        DialogueLine("zundamon", "で、信用倍率って何なのだ？"),
        DialogueLine("metan", "信用買い残を信用売り残で割った数値です。"),
        DialogueLine("metan", f"現在の信用倍率は{cr:.2f}倍。過去データと比較するとパーセンタイルは{percentile:.0f}パーセントです。"),
    ])

    # パーセンタイル解説
    if percentile <= 20:
        lines.append(DialogueLine("zundamon", "パーセンタイル20以下！かなり低いのだ！"))
        lines.append(DialogueLine("metan", "そうですね。過去データの中でも下位20パーセントに入る悲観的な水準です。"))
    elif percentile <= 40:
        lines.append(DialogueLine("zundamon", "やや低めなのだ？"))
        lines.append(DialogueLine("metan", "平均より悲観寄りですね。売り手が多い状況です。"))
    elif percentile <= 60:
        lines.append(DialogueLine("zundamon", "真ん中あたりなのだ！"))
        lines.append(DialogueLine("metan", "過去の平均的な水準ですね。特に偏りはありません。"))
    elif percentile <= 80:
        lines.append(DialogueLine("zundamon", "やや高めなのだ？"))
        lines.append(DialogueLine("metan", "楽観寄りですね。買い手が多い状況です。"))
    else:
        lines.append(DialogueLine("zundamon", "パーセンタイル80超え！かなり高いのだ！"))
        lines.append(DialogueLine("metan", "過去データの中でも上位20パーセントに入る強気な水準です。"))

    # === 評価損益の推定解説 ===
    lines.extend([
        DialogueLine("zundamon", "買い残が多いと何がわかるのだ？"),
        DialogueLine("metan", "実は、信用買いをしている人たちの含み損益が推定できるんです。"),
        DialogueLine("zundamon", "え！そんなことわかるのだ？"),
    ])

    # 買い残の評価損益状況を推定（倍率から推測）
    if cr > 2.0:
        lines.extend([
            DialogueLine("metan", "信用倍率が2倍を超えているということは、買い残が売り残の2倍以上あるということ。"),
            DialogueLine("metan", "多くの個人投資家が強気で買いポジションを持っています。"),
            DialogueLine("zundamon", "みんな儲かってるのだ？"),
            DialogueLine("metan", "相場が上がっていれば含み益ですが、ここから下がると一斉に含み損になるリスクがあります。"),
            DialogueLine("zundamon", "なるほど。高値掴みに注意なのだ！"),
        ])
    elif cr > 1.5:
        lines.extend([
            DialogueLine("metan", "買い手優勢の状況ですね。"),
            DialogueLine("metan", "相場が上がれば買い残は含み益、下がれば含み損になります。"),
            DialogueLine("zundamon", "上がってほしいのだ！"),
        ])
    elif cr > 1.0:
        lines.extend([
            DialogueLine("metan", "買い手と売り手がほぼ拮抗しています。"),
            DialogueLine("zundamon", "どっちが勝つかわからないのだ。"),
            DialogueLine("metan", "そうですね。次の材料で方向感が決まる局面です。"),
        ])
    else:
        lines.extend([
            DialogueLine("metan", "信用倍率1倍未満は、売り残が買い残より多い状況です。"),
            DialogueLine("zundamon", "みんな下がると思ってるのだ？"),
            DialogueLine("metan", "悲観的な見方が多いですね。ただし、売り残は将来の買い戻し需要でもあります。"),
            DialogueLine("zundamon", "踏み上げってやつなのだ？"),
            DialogueLine("metan", "その通り。相場が上がり始めると、売り方の買い戻しで上昇が加速することがあります。"),
        ])

    # === AIコメント ===
    if cached:
        lines.extend([
            DialogueLine("metan", "それでは臥龍ちゃんの分析です。"),
            DialogueLine("metan", cached),
        ])

    lines.append(DialogueLine("zundamon", "勉強になるのだ！"))

    return lines


def generate_volatility() -> list[DialogueLine]:
    """ボラティリティ分析"""
    cached = fetch_ai_comment("volatility")

    if cached:
        return [
            DialogueLine("metan", "続いてボラティリティ、つまり値動きの荒さを見ていきます。"),
            DialogueLine("zundamon", "ボラティリティ？難しい言葉なのだ。"),
            DialogueLine("metan", "簡単に言うと、相場がどれくらい激しく動いているかです。"),
            DialogueLine("zundamon", "ジェットコースターみたいなものなのだ？"),
            DialogueLine("metan", "そんなイメージですね。"),
            DialogueLine("metan", cached),
            DialogueLine("zundamon", "わかったのだ！"),
        ]

    # フォールバック
    data = fetch_data_json()
    if len(data) < 6:
        return [
            DialogueLine("metan", "ボラティリティのデータを収集中です。"),
            DialogueLine("zundamon", "待つのだ！"),
        ]

    prices = [d.get("Nikkei_Close", 0) for d in data[-6:] if d.get("Nikkei_Close")]
    if len(prices) < 5:
        return [
            DialogueLine("metan", "データが不足しています。"),
            DialogueLine("zundamon", "残念なのだ。"),
        ]

    returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
    vol = np.std(returns)
    vol_annualized = vol * np.sqrt(252) * 100

    if vol > 0.02:
        return [
            DialogueLine("metan", "ボラティリティを確認しましょう。"),
            DialogueLine("zundamon", "今の相場は荒れてるのだ？"),
            DialogueLine("metan", f"5日間のボラティリティは年率換算で{vol_annualized:.1f}パーセント。嵐の予感です。"),
            DialogueLine("zundamon", f"ひええ！荒れてるのだ！"),
            DialogueLine("metan", f"このような局面では慎重な立ち回りが重要です。"),
            DialogueLine("zundamon", f"無理しないのだ！"),
        ]
    elif vol > 0.01:
        return [
            DialogueLine("metan", "ボラティリティを見ていきましょう。"),
            DialogueLine("zundamon", "波は高いのだ？"),
            DialogueLine("metan", f"年率{vol_annualized:.1f}パーセント。やや波が立っています。"),
            DialogueLine("zundamon", f"ちょっと注意が必要なのだ？"),
            DialogueLine("metan", f"方向性を見極める局面ですね。"),
        ]
    elif vol < 0.005:
        return [
            DialogueLine("metan", "ボラティリティの確認です。"),
            DialogueLine("zundamon", "今日は穏やかなのだ？"),
            DialogueLine("metan", f"年率{vol_annualized:.1f}パーセント。凪の状態です。"),
            DialogueLine("zundamon", f"平和なのだ！"),
            DialogueLine("metan", f"ただし、静けさの後には嵐が来ることも。油断は禁物です。"),
            DialogueLine("zundamon", f"気を抜かないのだ！"),
        ]
    else:
        return [
            DialogueLine("metan", "ボラティリティです。"),
            DialogueLine("zundamon", "どうなのだ？"),
            DialogueLine("metan", f"年率{vol_annualized:.1f}パーセント。波は穏やかです。"),
            DialogueLine("zundamon", f"落ち着いてるのだ！"),
            DialogueLine("metan", f"次の大きな動きを待つ局面ですね。"),
        ]


def generate_correlation() -> list[DialogueLine]:
    """グローバル相関分析"""
    cached = fetch_ai_comment("global-correlation")

    if cached:
        return [
            DialogueLine("metan", "次はグローバル市場の相関を見ていきます。"),
            DialogueLine("zundamon", "相関？世界の市場が関係してるのだ？"),
            DialogueLine("metan", "そうです。日経平均は米国株やドル円など、様々な市場と連動しています。"),
            DialogueLine("zundamon", "世界は繋がっているのだ！"),
            DialogueLine("metan", cached),
            DialogueLine("zundamon", "グローバルな視点が大事なのだ！"),
        ]

    # フォールバック
    corr = fetch_correlation_matrix()
    matrix = corr.get("matrix", [])

    if not matrix:
        return [
            DialogueLine("metan", "相関データを取得中です。"),
            DialogueLine("zundamon", "待つのだ！"),
        ]

    max_corr = 0
    has_negative = False
    for i in range(len(matrix)):
        for j in range(i+1, len(matrix[i])):
            c = abs(matrix[i][j])
            if c > max_corr:
                max_corr = c
            if matrix[i][j] < -0.3:
                has_negative = True

    if max_corr >= 0.7:
        return [
            DialogueLine("metan", "グローバル相関を確認しましょう。"),
            DialogueLine("zundamon", "世界の市場はどうなのだ？"),
            DialogueLine("metan", "市場の連動性が非常に高い局面です。"),
            DialogueLine("zundamon", "みんな同じ方向に動いてるのだ？"),
            DialogueLine("metan", "リスクオン・リスクオフの動きに注意が必要です。"),
            DialogueLine("zundamon", "一緒に上がって一緒に下がるのだ！"),
        ]
    elif max_corr >= 0.5:
        if has_negative:
            return [
                DialogueLine("metan", "グローバル相関の確認です。"),
                DialogueLine("zundamon", "どんな感じなのだ？"),
                DialogueLine("metan", "正負の相関が混在しています。"),
                DialogueLine("zundamon", "バラバラなのだ？"),
                DialogueLine("metan", "分散投資の効果を活かせる局面かもしれません。"),
                DialogueLine("zundamon", "卵は一つのカゴに盛らないのだ！"),
            ]
        return [
            DialogueLine("metan", "グローバル相関です。"),
            DialogueLine("zundamon", "繋がりはどうなのだ？"),
            DialogueLine("metan", "主要市場は連動傾向にあります。"),
            DialogueLine("zundamon", "グローバルなトレンドを見るのだ！"),
            DialogueLine("metan", "その通りです。"),
        ]
    return [
        DialogueLine("metan", "グローバル相関を見ていきましょう。"),
        DialogueLine("zundamon", "世界との繋がりはどうなのだ？"),
        DialogueLine("metan", "相関は穏やかです。各市場が独自に動いています。"),
        DialogueLine("zundamon", "それぞれの事情があるのだ！"),
        DialogueLine("metan", "個別要因を見極めることが大切ですね。"),
    ]


def generate_option() -> list[DialogueLine]:
    """オプション建玉分析"""
    cached = fetch_ai_comment("option")

    if cached:
        return [
            DialogueLine("metan", "オプション市場の建玉分析です。"),
            DialogueLine("zundamon", "オプション？なんか難しそうなのだ。"),
            DialogueLine("metan", "簡単に言うと、プロ投資家がどこに壁を作っているかがわかります。"),
            DialogueLine("zundamon", "壁？"),
            DialogueLine("metan", "上値の抵抗線や下値の支持線のことです。"),
            DialogueLine("zundamon", "なるほどなのだ！"),
            DialogueLine("metan", cached),
            DialogueLine("zundamon", "プロの動きは参考になるのだ！"),
        ]

    # フォールバック
    oi = fetch_option_oi()
    positions = oi.get("positions", [])
    anomalies = oi.get("anomalies", [])
    atm_price = oi.get("atmPrice", 0)

    if not positions or atm_price == 0:
        return [
            DialogueLine("metan", "オプション建玉データを取得中です。"),
            DialogueLine("zundamon", "待つのだ！"),
        ]

    puts = [p for p in positions if p.get("type") == "PUT" and p.get("openInterest", 0) > 0]
    calls = [p for p in positions if p.get("type") == "CALL" and p.get("openInterest", 0) > 0]

    if not puts or not calls:
        return [
            DialogueLine("metan", "建玉データを分析中です。"),
            DialogueLine("zundamon", "頑張るのだ！"),
        ]

    top_puts = sorted(puts, key=lambda x: x.get("openInterest", 0), reverse=True)[:3]
    top_calls = sorted(calls, key=lambda x: x.get("openInterest", 0), reverse=True)[:3]
    top_put = top_puts[0]
    top_call = top_calls[0]

    put_distance = atm_price - top_put.get("strike", 0)
    call_distance = top_call.get("strike", 0) - atm_price

    lines = [
        DialogueLine("metan", "オプション建玉を分析していきましょう。"),
        DialogueLine("zundamon", "プロはどこを見てるのだ？"),
    ]

    if put_distance <= 3000 and call_distance <= 3000:
        lines.extend([
            DialogueLine("metan", f"プット{top_put['strike']:,}円とコール{top_call['strike']:,}円が近接しています。"),
            DialogueLine("zundamon", "上も下も壁があるのだ？"),
            DialogueLine("metan", "拮抗状態ですね。レンジ相場を示唆しています。"),
        ])
    elif put_distance <= 3000:
        lines.extend([
            DialogueLine("metan", f"プット{top_put['strike']:,}円に大きな建玉があります。"),
            DialogueLine("zundamon", "下値は守られてるのだ？"),
            DialogueLine("metan", "下値支持線として機能する可能性があります。"),
        ])
    elif call_distance <= 3000:
        lines.extend([
            DialogueLine("metan", f"コール{top_call['strike']:,}円に建玉が集中しています。"),
            DialogueLine("zundamon", "上は重いのだ？"),
            DialogueLine("metan", "上値抵抗線になる可能性がありますね。"),
        ])

    if anomalies:
        lines.extend([
            DialogueLine("metan", f"異常検知が{len(anomalies)}件あります。"),
            DialogueLine("zundamon", "異常！？大丈夫なのだ？"),
            DialogueLine("metan", "大きな値動きに警戒が必要です。"),
        ])
    else:
        lines.extend([
            DialogueLine("metan", "異常検知はありません。データは安定しています。"),
            DialogueLine("zundamon", "安心なのだ！"),
        ])

    return lines


def generate_narrative() -> list[DialogueLine]:
    """総合考察"""
    metrics = fetch_integrated_metrics()
    regime = metrics.get("regime", {}).get("latest", {}).get("regime", "range")
    cr = metrics.get("credit_ratio", {}).get("latest", {}).get("value", 1.2)

    lines = [
        DialogueLine("metan", "最後に、総合的な市場判断をお伝えします。"),
        DialogueLine("zundamon", "まとめなのだ！"),
    ]

    if regime == "bull":
        if cr > 2.0:
            lines.extend([
                DialogueLine("metan", "現在は強気相場ですが、楽観が過剰になっています。"),
                DialogueLine("zundamon", "調子に乗りすぎなのだ？"),
                DialogueLine("metan", "高値掴みには十分注意してください。"),
                DialogueLine("zundamon", "気をつけるのだ！"),
            ])
        elif cr < 1.0:
            lines.extend([
                DialogueLine("metan", "上昇トレンドの中に悲観が見られます。"),
                DialogueLine("zundamon", "上がってるのにみんな怖がってるのだ？"),
                DialogueLine("metan", "逆張り派にとっては好機かもしれません。"),
                DialogueLine("zundamon", "チャンスかもなのだ！"),
            ])
        else:
            lines.extend([
                DialogueLine("metan", "順風満帆な相場環境です。"),
                DialogueLine("zundamon", "やったのだ！いい感じなのだ！"),
                DialogueLine("metan", "ただし、驕れば足をすくわれます。謙虚にいきましょう。"),
                DialogueLine("zundamon", "油断大敵なのだ！"),
            ])
    elif regime == "bear":
        if cr < 1.0:
            lines.extend([
                DialogueLine("metan", "悲観の極みに達しています。"),
                DialogueLine("zundamon", "みんな落ち込んでるのだ。"),
                DialogueLine("metan", "歴史的には、このような局面が反転の好機になることも。"),
                DialogueLine("zundamon", "夜明け前が一番暗いのだ？"),
                DialogueLine("metan", "そうですね。ただし、慎重に。"),
            ])
        elif cr > 2.0:
            lines.extend([
                DialogueLine("metan", "下落相場なのに楽観が見られます。危険な兆候です。"),
                DialogueLine("zundamon", "えっ、それってまずいのだ？"),
                DialogueLine("metan", "はい。さらなる下落に注意が必要です。"),
                DialogueLine("zundamon", "気をつけるのだ。"),
            ])
        else:
            lines.extend([
                DialogueLine("metan", "冬の時代が続いています。"),
                DialogueLine("zundamon", "寒いのだ。"),
                DialogueLine("metan", "耐える覚悟と、次の春に備える準備が必要です。"),
                DialogueLine("zundamon", "いつか春は来るのだ！"),
            ])
    else:
        lines.extend([
            DialogueLine("metan", "現在は膠着状態、レンジ相場です。"),
            DialogueLine("zundamon", "動かないのだ？"),
            DialogueLine("metan", "次の大きな動きを待つ局面ですね。"),
            DialogueLine("zundamon", "じっと待つのだ！"),
        ])

    return lines


def generate_closing() -> list[DialogueLine]:
    """クロージング"""
    return [
        DialogueLine("metan", "以上、八門遁甲マーケットラジオでした。"),
        DialogueLine("zundamon", "今日も勉強になったのだ！"),
        DialogueLine("metan", "投資は自己責任で、慎重な判断をお願いします。"),
        DialogueLine("zundamon", "ちゃんと自分で考えるのだ！"),
        DialogueLine("metan", "それでは、また次回お会いしましょう。"),
        DialogueLine("zundamon", "ばいばいなのだ！"),
    ]


# コーナー定義
CORNERS = [
    ("opening", generate_opening),
    ("market-summary", None),  # 特別処理
    ("market-analysis", None),  # 特別処理: 市場要因分析
    ("sentiment", generate_sentiment),
    ("volatility", generate_volatility),
    ("correlation", generate_correlation),
    ("option", generate_option),
    ("narrative", generate_narrative),
    ("closing", generate_closing),
]


def generate_corner_audio(dialogues: list[DialogueLine], output_path: Path):
    """対話を音声化して1つのMP3に結合"""
    audio_segments = []

    for line in dialogues:
        # テキスト正規化
        text_clean = normalize_for_tts(line.text)
        # 絵文字を除去
        text_clean = re.sub(r'[\U0001F300-\U0001F9FF]', '', text_clean)

        if not text_clean.strip():
            continue

        print(f"    [{line.speaker}] {text_clean[:40]}...")

        speaker_id = SPEAKERS[line.speaker]
        wav_bytes = synthesize_voicevox(text_clean, speaker_id)
        segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        audio_segments.append(segment)
        # 話者間に200msの無音（自然な間）
        audio_segments.append(AudioSegment.silent(duration=200))

    # 結合
    combined = AudioSegment.empty()
    for seg in audio_segments:
        combined += seg

    # MP3エクスポート
    combined.export(output_path, format="mp3", bitrate="192k")


def main():
    parser = argparse.ArgumentParser(description='Generate Garyu Radio audio (VOICEVOX duo)')
    parser.add_argument('--output-dir', type=str, help='Output directory for audio files')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    print("=== 八門遁甲マーケットラジオ音声生成開始 ===")
    print(f"  VOICEVOX URL: {VOICEVOX_URL}")
    print(f"  Output dir: {output_dir}")

    # VOICEVOX接続確認
    try:
        version_resp = requests.get(f"{VOICEVOX_URL}/version", timeout=5)
        print(f"  VOICEVOX version: {version_resp.text}")
    except Exception as e:
        print(f"  ERROR: VOICEVOX connection failed: {e}")
        print("  Please start VOICEVOX engine first.")
        return

    # 市場データ・ニュース・天気を事前取得
    print("\nFetching market data, news, and weather...")
    market_data = fetch_market_live()
    news = fetch_news()
    weather = fetch_weather()

    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # マニフェスト初期化
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "voice": "VOICEVOX:ずんだもん+四国めたん",
        "files": []
    }

    # 各コーナーの音声を生成
    for corner_id, dialogue_func in CORNERS:
        print(f"\nGenerating {corner_id}...")

        # 特別処理が必要なコーナー
        if corner_id == "opening":
            dialogues = generate_opening(weather)
        elif corner_id == "market-summary":
            dialogues = generate_market_summary(market_data, news)
        elif corner_id == "market-analysis":
            dialogues = generate_market_analysis(market_data)
        else:
            dialogues = dialogue_func()

        output_path = output_dir / f"{corner_id}.mp3"
        generate_corner_audio(dialogues, output_path)

        file_size = output_path.stat().st_size
        manifest["files"].append({
            "id": corner_id,
            "path": f"/audio/{corner_id}.mp3",
            "size": file_size
        })
        print(f"  Saved: {output_path} ({file_size:,} bytes)")

    # マニフェスト保存
    manifest_path = output_dir / "audio-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\nManifest saved: {manifest_path}")
    print("=== 音声生成完了 ===")


if __name__ == "__main__":
    main()
