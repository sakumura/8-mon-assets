"""
チュートリアル音声生成スクリプト
日経平均八門遁甲システムの使い方を解説する固定音声を生成

音声モデル: つくよみちゃん Style-BERT-VITS2
- モデル提供: ayousanz (https://huggingface.co/ayousanz/tsukuyomi-chan-style-bert-vits2-model)
- キャラクター: つくよみちゃん (https://tyc.rei-yumesaki.net/)
- ライセンス: つくよみちゃんキャラクターライセンス
"""
import os
import tempfile
from pathlib import Path

# Style-BERT-VITS2
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
from huggingface_hub import hf_hub_download

# pydubでWAV→MP3変換
from pydub import AudioSegment
import numpy as np
import scipy.io.wavfile as wavfile

# パス設定
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
MODEL_DIR = SCRIPT_DIR / "model_assets"

# つくよみちゃん モデルのHugging Face情報
HF_REPO_ID = "ayousanz/tsukuyomi-chan-style-bert-vits2-model"
MODEL_FILES = {
    "model": "tsukuyomi-chan_e200_s5200.safetensors",
    "config": "config.json",
    "style": "style_vectors.npy",
}

# チュートリアル台本（固定）
TUTORIAL_TEXT = """
日経平均八門遁甲システムへようこそ。わらわ、AI臥龍が案内するぞ。

このシステムでは、日経平均の動きを8つの視点から分析しておる。

画面上段には3つの球体が並んでおる。
左の「恐怖と強欲の球体」は、信用倍率から市場センチメントを映し出す。
中央の「波動ホログラム」は、ボラティリティの高低を波の形で表現しておる。
右の「相関ネットワーク」は、世界市場との連動性を可視化しておる。

その下には「オプション建玉ホロテーブル」がある。
プットとコールの建玉分布から、相場の壁と支えを読み解けるぞ。

右のパネルには市場ニュースとカレンダーが表示される。

画面左上の「臥龍ラジオ」ボタンを押すと、わらわが最新の市場解説を音声でお届けする。

それでは、良きトレードを。
"""


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


def generate_audio(model: TTSModel, text: str, output_path: Path):
    """Style-BERT-VITS2でテキストを音声化してMP3保存"""
    print(f"Generating audio for text ({len(text)} chars)...")

    # 音質改善パラメータ
    sr, audio = model.infer(
        text=text,
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
        print(f"  Saved: {output_path} ({output_path.stat().st_size:,} bytes)")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    print("=== チュートリアル音声生成開始 ===")

    # TTSモデル初期化
    model = init_tts_model()

    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # チュートリアル音声生成
    output_path = OUTPUT_DIR / "tutorial.mp3"
    generate_audio(model, TUTORIAL_TEXT.strip(), output_path)

    print("=== チュートリアル音声生成完了 ===")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
