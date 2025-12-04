"""
臥龍ラジオ音声生成スクリプト（8-mon-assets版）
Style-BERT-VITS2 (つくよみちゃん) を使用して市場解説音声を生成

音声モデル: つくよみちゃん Style-BERT-VITS2
- モデル提供: ayousanz (https://huggingface.co/ayousanz/tsukuyomi-chan-style-bert-vits2-model)
- キャラクター: つくよみちゃん (https://tyc.rei-yumesaki.net/)
- ライセンス: つくよみちゃんキャラクターライセンス
"""
import os
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import argparse

# Style-BERT-VITS2
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
from huggingface_hub import hf_hub_download

# pydubでWAV→MP3変換
from pydub import AudioSegment
import numpy as np
import scipy.io.wavfile as wavfile
import urllib.request
import re

# パス設定
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
MODEL_DIR = SCRIPT_DIR / "model_assets"
AI_CACHE_DIR = SCRIPT_DIR / "ai_cache"

# つくよみちゃん モデルのHugging Face情報
HF_REPO_ID = "ayousanz/tsukuyomi-chan-style-bert-vits2-model"
MODEL_FILES = {
    "model": "tsukuyomi-chan_e200_s5200.safetensors",
    "config": "config.json",
    "style": "style_vectors.npy",
}


def download_model_files():
    """Hugging Faceからモデルファイルをダウンロード"""
    print("Downloading model files from Hugging Face...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    local_files = {}
    for key, remote_path in MODEL_FILES.items():
        print(f"  Downloading {remote_path}...")
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=remote_path,
            local_dir=MODEL_DIR
        )
        local_files[key] = Path(local_path)
        print(f"    -> {local_path}")

    return local_files


def init_tts_model():
    """TTSモデルを初期化"""
    print("Loading BERT model...")
    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

    # モデルファイルをダウンロード
    files = download_model_files()

    print("Loading TTS model (tsukuyomi-chan)...")
    model = TTSModel(
        model_path=files["model"],
        config_path=files["config"],
        style_vec_path=files["style"],
        device="cpu"
    )
    print("Models loaded successfully!")
    return model


def get_jst_time() -> str:
    """JST現在時刻を取得"""
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).strftime("%H時%M分")


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
    }
    for original, reading in replacements.items():
        text = text.replace(original, reading)
    return text


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
            headers={"User-Agent": "GaryuRadio/1.0"}
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

        req = urllib.request.Request(url, headers={"User-Agent": "GaryuRadio/1.0"})
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


def generate_opening() -> str:
    """オープニングテキスト生成"""
    time = get_jst_time()
    return f"臥龍ラジオ、開演じゃ。現在時刻は{time}。本日の市場を読み解いていくぞ。"


def generate_market_summary(market_data: dict = None, news: list[dict] = None) -> str:
    """市況サマリー生成（リアルタイムAPI使用）"""
    try:
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
                return "市場データを確認中じゃ..."
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

        # セッションで現物/先物を切り替え
        if session == "futures":
            market_type = "日経先物"
        else:
            market_type = "日経平均"

        # 前日比テキスト
        if change != 0:
            sign = "プラス" if change >= 0 else "マイナス"
            change_str = format_price_for_tts(abs(change))
            text = f"{market_type}は現在{price_str}円。前日比{sign}{change_str}円、{sign}{abs(change_pct):.2f}パーセントで推移しておる。"
        else:
            text = f"{market_type}は現在{price_str}円じゃ。"

        # ニュース追加
        if news:
            text += "続いて、本日の注目ニュースじゃ。"
            for i, item in enumerate(news, 1):
                headline = item.get("headline", "")
                text += f"{i}件目、{headline}。"

        return text
    except Exception as e:
        print(f"  Warning: market-summary error: {e}")
        return "市場データを確認中じゃ..."


def generate_sentiment() -> str:
    """AIキャッシュまたはintegrated-metricsからセンチメント生成"""
    # AIコメントを取得（feargreed = センチメント分析）
    cached = fetch_ai_comment("feargreed")
    if cached:
        return cached

    # フォールバック - より詳細な分析
    try:
        metrics = fetch_integrated_metrics()
        cr_data = metrics.get("credit_ratio", {}).get("latest", {})
        cr = cr_data.get("value", 1.2)

        # Fear & Greed指数の計算（信用倍率ベース）
        if cr < 0.8:
            comment = f"信用倍率{cr:.2f}倍。極度の恐怖が市場を支配しておる。大衆が逃げ惑うとき、逆張りの好機が訪れることもある。されど、落ちるナイフを掴むなかれ。"
        elif cr < 1.0:
            comment = f"信用倍率{cr:.2f}倍。恐怖が広がっておる。売り手優勢の局面じゃ。悲観の極みは反転の兆しとなることもあるが、時期尚早の判断は禁物じゃ。"
        elif cr < 1.5:
            comment = f"信用倍率{cr:.2f}倍。センチメントは均衡状態にある。買い手と売り手が拮抗しておる。次の方向性を見極めるべき局面じゃ。"
        elif cr < 2.0:
            comment = f"信用倍率{cr:.2f}倍。楽観ムードが広がりつつある。買い手優勢の局面じゃが、過熱感には注意が必要じゃ。"
        elif cr < 3.0:
            comment = f"信用倍率{cr:.2f}倍。強欲が市場を支配しておる。大衆が熱狂するとき、賢者は慎重になる。高値掴みに警戒せよ。"
        else:
            comment = f"信用倍率{cr:.2f}倍。極度の強欲状態じゃ。歴史的に見て、このような過熱相場は調整を招くことが多い。利益確定を検討すべき局面かもしれぬ。"

        return comment
    except Exception as e:
        print(f"  Warning: sentiment error: {e}")
        return "センチメントを分析中じゃ..."


def generate_volatility() -> str:
    """AIコメントまたはdata.jsonからボラティリティコメント生成"""
    # AIコメントを取得
    cached = fetch_ai_comment("volatility")
    if cached:
        return cached

    # フォールバック - 詳細分析
    try:
        data = fetch_data_json()
        if len(data) < 6:
            return "波の動きを見極めよ。"

        prices = [d.get("Nikkei_Close", 0) for d in data[-6:] if d.get("Nikkei_Close")]
        if len(prices) < 6:
            return "波の動きを見極めよ。"

        # 日次リターンとボラティリティ計算
        returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
        vol = np.std(returns)
        vol_annualized = vol * np.sqrt(252) * 100  # 年率換算（%）

        # 最大・最小リターン
        max_return = max(returns) * 100
        min_return = min(returns) * 100

        if vol > 0.02:
            comment = f"5日間のボラティリティは年率換算で{vol_annualized:.1f}パーセント。嵐の予感じゃ。最大変動{max_return:+.1f}パーセント、最小変動{min_return:+.1f}パーセント。荒れ相場では慎重な立ち回りが肝要じゃ。"
        elif vol > 0.01:
            comment = f"5日間のボラティリティは年率{vol_annualized:.1f}パーセント。やや波が立っておる。最大{max_return:+.1f}パーセント、最小{min_return:+.1f}パーセントの変動を記録。方向性を見極める局面じゃ。"
        elif vol < 0.005:
            comment = f"5日間のボラティリティは年率{vol_annualized:.1f}パーセント。凪の時じゃ。静けさの後には嵐が来ることもある。油断するなかれ。"
        else:
            comment = f"5日間のボラティリティは年率{vol_annualized:.1f}パーセント。波は穏やかじゃ。次の大きな動きを待つべし。"

        return comment
    except Exception as e:
        print(f"  Warning: volatility error: {e}")
        return "ボラティリティを観測中じゃ..."


def generate_correlation() -> str:
    """AIコメントまたはcorrelation-matrixからグローバル相関コメント生成"""
    # AIコメントを取得
    cached = fetch_ai_comment("global-correlation")
    if cached:
        return cached

    # フォールバック
    try:
        corr = fetch_correlation_matrix()
        matrix = corr.get("matrix", [])
        if not matrix:
            return "世界市場の繋がりを見よ。"

        # 最大相関を計算
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
            return "市場の連動性が高い局面じゃ。リスクオンオフの動きに注意せよ。"
        elif max_corr >= 0.5:
            if has_negative:
                return "正負の相関が混在しておる。分散効果を活かす好機かもしれぬ。"
            return "主要市場は連動傾向にあり。グローバルなトレンドを見よ。"
        return "相関は穏やかじゃ。各市場の個別要因を見極めよ。"
    except Exception as e:
        print(f"  Warning: correlation error: {e}")
        return "世界市場の繋がりを見よ。"


def generate_option() -> str:
    """AIコメントまたはoption-oi.jsonからオプション分析コメント生成"""
    # AIコメントを取得（最優先）
    cached = fetch_ai_comment("option")
    if cached:
        return cached

    # フォールバック
    try:
        oi = fetch_option_oi()
        positions = oi.get("positions", [])
        anomalies = oi.get("anomalies", [])
        atm_price = oi.get("atmPrice", 0)

        if not positions or atm_price == 0:
            return "建玉データを分析中じゃ。"

        # PUT/CALLを分離
        puts = [p for p in positions if p.get("type") == "PUT" and p.get("openInterest", 0) > 0]
        calls = [p for p in positions if p.get("type") == "CALL" and p.get("openInterest", 0) > 0]

        if not puts or not calls:
            return "建玉データを分析中じゃ。"

        # 建玉上位を取得
        top_puts = sorted(puts, key=lambda x: x.get("openInterest", 0), reverse=True)[:3]
        top_calls = sorted(calls, key=lambda x: x.get("openInterest", 0), reverse=True)[:3]

        top_put = top_puts[0]
        top_call = top_calls[0]

        # ATMからの距離
        put_distance = atm_price - top_put.get("strike", 0)
        call_distance = top_call.get("strike", 0) - atm_price

        comment_parts = []

        # 近接ゾーン分析
        if put_distance <= 3000 and call_distance <= 3000:
            comment_parts.append(
                f"近接ゾーンではプット{top_put['strike']:,}円とコール{top_call['strike']:,}円が拮抗しておる。"
            )
        elif put_distance <= 3000:
            comment_parts.append(
                f"プット{top_put['strike']:,}円が下値を支える構えじゃ。"
            )
        elif call_distance <= 3000:
            comment_parts.append(
                f"コール{top_call['strike']:,}円が上値を抑えておる。"
            )

        # FOTM分析
        if put_distance > 5000:
            comment_parts.append(
                f"遠方のプット{top_put['strike']:,}円に大量建玉あり。"
            )
        if call_distance > 5000:
            comment_parts.append(
                f"遠方のコール{top_call['strike']:,}円に建玉集中。"
            )

        # 異常検知
        if anomalies:
            comment_parts.append(
                f"異常検知{len(anomalies)}件。激しい値動きに警戒せよ。"
            )
        else:
            comment_parts.append("異常検知なし。データは安定しておる。")

        return "".join(comment_parts) if comment_parts else "オプション建玉の動きを見定めよ。"

    except Exception as e:
        print(f"  Warning: option error: {e}")
        return "オプションの動きを読み解くぞ。"


def generate_narrative() -> str:
    """integrated-metricsから総合考察生成"""
    try:
        metrics = fetch_integrated_metrics()
        regime = metrics.get("regime", {}).get("latest", {}).get("regime", "range")
        cr = metrics.get("credit_ratio", {}).get("latest", {}).get("value", 1.2)

        if regime == "bull":
            if cr > 2.0:
                return "強気相場だが楽観過剰じゃ。高値掴みに注意せよ。"
            if cr < 1.0:
                return "上昇トレンドに悲観あり。逆張り派の好機かもしれぬ。"
            return "順風満帆。されど驕るなかれ。"
        if regime == "bear":
            if cr < 1.0:
                return "悲観の極み。歴史は反転の好機を示すが。"
            if cr > 2.0:
                return "下落相場に楽観あり。危うい兆候じゃ。"
            return "冬の時代。耐える覚悟を。"
        return "膠着状態じゃ。次の一手を待て。"
    except Exception as e:
        print(f"  Warning: narrative error: {e}")
        return "市場を俯瞰しておる..."


def generate_closing() -> str:
    """クロージングテキスト生成"""
    return "以上、臥龍ラジオでした。投資は自己責任で。引き続きご注意を。"


# コーナー定義
CORNERS = [
    ("opening", generate_opening),
    ("market-summary", generate_market_summary),
    ("sentiment", generate_sentiment),
    ("volatility", generate_volatility),
    ("correlation", generate_correlation),
    ("option", generate_option),
    ("narrative", generate_narrative),
    ("closing", generate_closing),
]


def generate_audio(model: TTSModel, text: str, output_path: Path):
    """Style-BERT-VITS2でテキストを音声化してMP3保存"""
    # 用語読み替え処理を適用
    text_normalized = normalize_for_tts(text)
    # 絵文字を除去（TTSが対応していない場合のため）
    text_clean = re.sub(r'[\U0001F300-\U0001F9FF]', '', text_normalized)

    # 音質改善パラメータ
    sr, audio = model.infer(
        text=text_clean,
        noise=0.4,        # ノイズ軽減
        noise_w=0.6,      # 発話安定化
        sdp_ratio=0.2,    # 話速安定化
        length=1.0,       # 話速（1.0=通常）
    )

    # 一時WAVファイルに保存してからMP3変換
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # 音声の正規化（クリッピング防止）
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95  # ピークを95%に正規化

        # numpy配列をint16に変換
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(tmp_path, sr, audio_int16)

        # WAV→MP3変換（高品質設定）
        sound = AudioSegment.from_wav(tmp_path)
        sound.export(output_path, format="mp3", bitrate="192k")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description='Generate Garyu Radio audio')
    parser.add_argument('--output-dir', type=str, help='Output directory for audio files')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    print("=== 臥龍ラジオ音声生成開始 ===")

    # 市場データとニュースを事前取得
    print("\nFetching market data and news...")
    market_data = fetch_market_live()
    news = fetch_news()

    # TTSモデル初期化
    model = init_tts_model()

    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # マニフェスト初期化
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "files": []
    }

    # 各コーナーの音声を生成
    for corner_id, text_func in CORNERS:
        print(f"\nGenerating {corner_id}...")
        # market-summaryは特別処理（市場データとニュースを渡す）
        if corner_id == "market-summary":
            text = generate_market_summary(market_data, news)
        else:
            text = text_func()
        print(f"  Text: {text[:80]}...")

        output_path = output_dir / f"{corner_id}.mp3"
        generate_audio(model, text, output_path)

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
