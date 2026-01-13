"""
AI臥龍日記 音声生成スクリプト（8-mon-assets版）
VOICEVOXを使用して日記を音声化

音声: VOICEVOX:四国めたん
- VOICEVOX: https://voicevox.hiroshiba.jp/
- キャラクター利用規約: https://zunko.jp/con_ongen_kiyaku.html
"""
import os
import io
import re
import sys
import json
from datetime import datetime
from pathlib import Path
import argparse

import requests
from pydub import AudioSegment

# VOICEVOX設定
VOICEVOX_URL = os.environ.get("VOICEVOX_URL", "http://localhost:50021")
SPEAKER_ID = 2  # 四国めたん（ノーマル）

# 8-mon本体からの日記取得URL（ローカルまたは本番）
DIARY_BASE_URL = os.environ.get("DIARY_BASE_URL", "https://8-mon.com/diary")


def synthesize_voicevox(text: str, speaker_id: int = SPEAKER_ID) -> bytes:
    """VOICEVOXで音声合成"""
    query_resp = requests.post(
        f"{VOICEVOX_URL}/audio_query",
        params={"text": text, "speaker": speaker_id},
        timeout=30
    )
    query_resp.raise_for_status()
    query = query_resp.json()

    synth_resp = requests.post(
        f"{VOICEVOX_URL}/synthesis",
        params={"speaker": speaker_id},
        json=query,
        timeout=60
    )
    synth_resp.raise_for_status()
    return synth_resp.content


def normalize_for_tts(text: str) -> str:
    """TTS用にテキストを正規化（読み替え）"""
    replacements = {
        # 英語略語
        "BTC": "ビットコイン",
        "NASDAQ": "ナスダック",
        "USD/JPY": "ドル円",
        "GOLD": "ゴールド",
        "VIX": "ビックス",
        "ETF": "イーティーエフ",
        "ATM IV": "エーティーエム アイブイ",
        "IV": "アイブイ",
        "HV": "エイチブイ",
        "1570": "イチゴーナナゼロ",
        "N5": "エヌファイブ",
        "N220": "エヌニーニーゼロ",
        "LONG": "ロング",
        "SHORT": "ショート",
        # 日本語読み
        "日経225": "ニッケイニーニーゴ",
        "日経平均": "にっけいへいきん",
        "臥龍": "がりょう",
        "値嵩": "ねがさ",
        "信用倍率": "しんようばいりつ",
        "貸株金利": "かしかぶきんり",
        "建玉": "たてぎょく",
        "寄与率": "きよりつ",
        "逆張り": "ぎゃくばり",
        "順張り": "じゅんばり",
        "買越": "かいこし",
        "売越": "うりこし",
        # 三国志キャラ
        "諸葛亮": "しょかつりょう",
        "孔明": "こうめい",
        "諸君": "しょくん",
        # 記号
        "━": "", "■": "", "▼": "", "【": "", "】": "",
        "📊": "", "📅": "", "##": "",
        "→": "、", "…": "、", "|": "、", "※": "、なお、",
        "**": "",  # マークダウン強調
    }
    for original, reading in replacements.items():
        text = text.replace(original, reading)

    # 絵文字除去
    text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)
    # 複数空白を単一に
    text = re.sub(r'\s+', ' ', text)
    # マークダウンリンク除去
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    return text.strip()


def fetch_diary_markdown(date: str) -> str:
    """8-monから日記Markdownを取得"""
    url = f"{DIARY_BASE_URL}/{date}.md"
    try:
        resp = requests.get(url, timeout=10)
        if resp.ok:
            return resp.text
    except Exception as e:
        print(f"Warning: Failed to fetch from URL: {e}")

    # ローカルファイルからフォールバック
    local_path = Path(__file__).parent.parent.parent / "8-mon" / "frontend" / "public" / "diary" / f"{date}.md"
    if local_path.exists():
        return local_path.read_text(encoding="utf-8")

    raise FileNotFoundError(f"Diary not found: {date}")


def fetch_diary_meta(date: str) -> dict:
    """8-monから日記メタデータを取得"""
    url = f"{DIARY_BASE_URL}/{date}.json"
    try:
        resp = requests.get(url, timeout=10)
        if resp.ok:
            return resp.json()
    except:
        pass

    # ローカルファイルからフォールバック
    local_path = Path(__file__).parent.parent.parent / "8-mon" / "frontend" / "public" / "diary" / f"{date}.json"
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))

    return {}


def parse_diary_markdown(content: str) -> dict:
    """日記Markdownをセクションに分解"""
    sections = {}
    current_section = "header"
    current_content = []

    for line in content.split("\n"):
        # ## セクションヘッダー
        if line.startswith("## "):
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = line.replace("## ", "").strip()
            current_content = []
        # # タイトル（スキップ）
        elif line.startswith("# "):
            continue
        else:
            current_content.append(line)

    if current_content:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def get_time_slot_greeting(time_slot: str) -> tuple[str, str]:
    """時刻スロットに応じた挨拶と締め"""
    greetings = {
        "morning": (
            "おはようございます。AI臥龍の朝の市場分析をお届けします。",
            "以上、AI臥龍の朝の分析でした。本日も良いトレードを。"
        ),
        "noon": (
            "こんにちは。AI臥龍の前場終了時点での分析をお届けします。",
            "以上、AI臥龍の前場分析でした。後場も注視していきましょう。"
        ),
        "evening": (
            "こんにちは。AI臥龍の大引け後の総括をお届けします。",
            "以上、AI臥龍の本日の総括でした。明日も市場の動きに注目していきましょう。"
        ),
        "night": (
            "こんばんは。AI臥龍の夜間市場分析をお届けします。",
            "以上、AI臥龍の夜間分析でした。おやすみなさい。"
        ),
    }
    return greetings.get(time_slot, greetings["morning"])


def diary_to_script(content: str, meta: dict) -> str:
    """日記Markdownを読み上げ原稿に変換"""
    sections = parse_diary_markdown(content)
    time_slot = meta.get("time_slot", "morning")
    date_str = meta.get("date", "")

    # 日付の読み上げ形式
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        date_readable = f"{dt.year}年{dt.month}月{dt.day}日、{weekdays[dt.weekday()]}曜日"
    except:
        date_readable = date_str

    greeting, closing = get_time_slot_greeting(time_slot)
    script_parts = [f"{greeting} 本日は{date_readable}です。"]

    # 価格情報
    price_info = meta.get("price", {})
    if price_info.get("current"):
        price = price_info["current"]
        change_pct = price_info.get("change_pct", 0)
        source = price_info.get("source", "")
        direction = "上昇" if change_pct >= 0 else "下落"
        script_parts.append(
            f"現在の日経平均{source}は{price:,.0f}円、前日比{abs(change_pct):.2f}パーセント{direction}しています。"
        )

    # 各セクション（刊によって異なるセクション名に対応）
    section_order = [
        # 共通
        "現在の相場位置",
        "本日の一言",
        # 朝刊
        "順張りシグナル分析",
        "逆張りシグナル分析",
        "注目の節目",
        "本日の展望",
        "予測検証",
        # 昼刊
        "前場の振り返り",
        "後場の展望",
        "注目ポイント",
        # 夕刊・夜刊
        "本日の総括",
        "明日の展望",
        "市場センチメント",
        "海外市場の見通し",
    ]

    for section_name in section_order:
        if section_name in sections:
            section_content = sections[section_name]
            # リスト項目を文章化
            lines = []
            for line in section_content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # マークダウンリスト項目
                if line.startswith("- "):
                    line = line[2:]
                # 注釈行はスキップ
                if line.startswith("※"):
                    continue
                lines.append(line)

            if lines:
                text = " ".join(lines)
                # セクション名を読み上げ
                if section_name == "本日の一言":
                    script_parts.append(f"最後に、本日の一言です。{text}")
                else:
                    script_parts.append(f"{section_name}です。{text}")

    script_parts.append(closing)
    script_parts.append("本レポートは情報提供を目的としており、投資助言ではありません。投資判断は自己責任でお願いいたします。")

    return "\n\n".join(script_parts)


def generate_audio_from_script(script: str, output_path: Path, pause_ms: int = 500):
    """原稿から音声を生成"""
    audio_segments = []
    paragraphs = [p.strip() for p in script.split("\n\n") if p.strip()]

    for i, paragraph in enumerate(paragraphs):
        sentences = re.split(r'(?<=[。！？])', paragraph)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            text_clean = normalize_for_tts(sentence)
            if not text_clean:
                continue

            # 長いテキストは分割
            if len(text_clean) > 100:
                parts = text_clean.split("、")
                for j, part in enumerate(parts):
                    if not part.strip():
                        continue
                    part_text = part.strip() + ("、" if j < len(parts) - 1 else "")
                    print(f"    [{i+1}] {part_text[:50]}...")
                    wav_bytes = synthesize_voicevox(part_text)
                    audio_segments.append(AudioSegment.from_wav(io.BytesIO(wav_bytes)))
                    audio_segments.append(AudioSegment.silent(duration=150))
            else:
                print(f"    [{i+1}] {text_clean[:50]}...")
                wav_bytes = synthesize_voicevox(text_clean)
                audio_segments.append(AudioSegment.from_wav(io.BytesIO(wav_bytes)))
                audio_segments.append(AudioSegment.silent(duration=300))

        # 段落間ポーズ
        audio_segments.append(AudioSegment.silent(duration=pause_ms))

    # 結合
    combined = AudioSegment.empty()
    for seg in audio_segments:
        combined += seg

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="mp3", bitrate="192k")
    return output_path


def update_manifest(output_dir: Path, date: str, time_slot: str, filename: str):
    """マニフェストファイルを更新"""
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"entries": [], "latest": {}}

    # エントリ更新
    entry = next((e for e in manifest["entries"] if e["date"] == date), None)
    if entry:
        if "audio" not in entry:
            entry["audio"] = {}
        entry["audio"][time_slot] = filename
    else:
        manifest["entries"].append({
            "date": date,
            "audio": {time_slot: filename}
        })

    # 日付でソート（新しい順）
    manifest["entries"].sort(key=lambda x: x["date"], reverse=True)

    # latest更新
    manifest["latest"][time_slot] = filename

    # 最新100件のみ保持
    manifest["entries"] = manifest["entries"][:100]

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Manifest updated: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate audio from AI臥龍 diary')
    parser.add_argument('--date', '-d', type=str, help='Diary date (YYYY-MM-DD). Default: today')
    parser.add_argument('--time-slot', '-t', type=str, choices=['morning', 'noon', 'evening', 'night'],
                        help='Time slot. Auto-detected from meta if not specified')
    parser.add_argument('--output-dir', '-o', type=str, default='output/diary',
                        help='Output directory for MP3 files')
    parser.add_argument('--local', '-l', type=str, help='Use local markdown file instead of fetching')
    args = parser.parse_args()

    # VOICEVOX接続確認
    try:
        version_resp = requests.get(f"{VOICEVOX_URL}/version", timeout=5)
        print(f"VOICEVOX version: {version_resp.text}")
    except Exception as e:
        print(f"ERROR: VOICEVOX connection failed: {e}")
        print(f"  Make sure VOICEVOX is running at {VOICEVOX_URL}")
        sys.exit(1)

    # 日付決定
    if args.date:
        date = args.date
    else:
        date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n=== Processing diary: {date} ===")

    # 日記取得
    try:
        if args.local:
            content = Path(args.local).read_text(encoding="utf-8")
            meta = {}
        else:
            content = fetch_diary_markdown(date)
            meta = fetch_diary_meta(date)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # タイムスロット決定
    time_slot = args.time_slot or meta.get("time_slot", "morning")
    meta["time_slot"] = time_slot
    meta["date"] = date

    print(f"  Time slot: {time_slot}")
    print(f"  Content length: {len(content)} chars")

    # 原稿生成
    script = diary_to_script(content, meta)
    print(f"  Script length: {len(script)} chars")

    # 音声生成
    output_dir = Path(args.output_dir)
    filename = f"{time_slot}_{date}.mp3"
    output_path = output_dir / filename

    print(f"  Generating audio...")
    generate_audio_from_script(script, output_path)

    file_size = output_path.stat().st_size
    print(f"  Output: {output_path} ({file_size:,} bytes)")

    # マニフェスト更新
    update_manifest(output_dir, date, time_slot, filename)

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
