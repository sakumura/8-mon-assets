"""
ニュースレター音声生成スクリプト（8-mon-assets版）
VOICEVOXを使用してニュースレターを一人読み形式で音声化

音声: VOICEVOX:四国めたん
- VOICEVOX: https://voicevox.hiroshiba.jp/
- キャラクター利用規約: https://zunko.jp/con_ongen_kiyaku.html
"""
import os
import io
import re
import sys
from datetime import datetime
from pathlib import Path
import argparse

import requests
from pydub import AudioSegment

# VOICEVOX設定
VOICEVOX_URL = os.environ.get("VOICEVOX_URL", "http://localhost:50021")
SPEAKER_ID = 2  # 四国めたん（ノーマル）


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
        # 日本語読み
        "日経225": "ニッケイニーニーゴ",
        "臥龍": "がりょう",
        "値嵩": "ねがさ",
        "信用倍率": "しんようばいりつ",
        "貸株金利": "かしかぶきんり",
        "建玉": "たてぎょく",
        "寄与率": "きよりつ",
        # 記号
        "━": "", "■": "", "▼": "", "【": "", "】": "",
        "📊": "", "📅": "",
        "→": "、", "…": "、", "|": "、", "※": "、なお、",
    }
    for original, reading in replacements.items():
        text = text.replace(original, reading)
    text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_newsletter_txt(txt_path: Path) -> dict:
    """ニュースレターTXTをセクション辞書に分解"""
    content = txt_path.read_text(encoding="utf-8")
    filename = txt_path.stem

    if filename.startswith("night_"):
        version = "night"
        date_str = filename.replace("night_", "")
    elif filename.startswith("morning_"):
        version = "morning"
        date_str = filename.replace("morning_", "")
    elif filename.startswith("lunch_"):
        version = "lunch"
        date_str = filename.replace("lunch_", "")
    else:
        version = "unknown"
        date_str = ""

    sections = {}
    current_section = "header"
    current_content = []

    for line in content.split("\n"):
        if line.startswith("■"):
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            match = re.match(r"■\s*(\d+)\.\s*(.+)", line)
            if match:
                current_section = f"{match.group(1)}_{match.group(2).strip()}"
            else:
                current_section = line.replace("■", "").strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections[current_section] = "\n".join(current_content).strip()

    return {"version": version, "date": date_str, "sections": sections, "raw_content": content}


def newsletter_to_script_night(parsed: dict) -> str:
    """夜版ニュースレターを一人読み原稿に変換"""
    sections = parsed["sections"]
    date_str = parsed["date"]

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        date_readable = f"{dt.month}月{dt.day}日、{weekdays[dt.weekday()]}曜日"
    except:
        date_readable = date_str

    script_parts = [f"こんばんは。八門遁甲ナイトレポートをお届けします。本日は{date_readable}です。"]

    # 各セクション処理
    if "1_本日の市場総括" in sections:
        content = sections["1_本日の市場総括"]
        regime_match = re.search(r"【レジーム】(.+?)(?:\n|$)", content)
        regime = regime_match.group(1).strip() if regime_match else ""
        narrative_match = re.search(r"【市場ナラティブ】\n(.+?)(?:\n\n|$)", content, re.DOTALL)
        narrative = narrative_match.group(1).strip() if narrative_match else ""
        script_parts.append(f"本日の市場総括です。{regime} {narrative}")

    if "2_信用倍率（1570 日経レバETF）" in sections:
        content = sections["2_信用倍率（1570 日経レバETF）"]
        lines = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("※")][:4]
        script_parts.append(f"続いて、1570日経レバイーティーエフの信用倍率です。{' '.join(lines)}")

    if "3_需給マトリックス（1570 日経レバETF）" in sections:
        content = sections["3_需給マトリックス（1570 日経レバETF）"]
        lines = [l.strip() for l in content.split("\n") if l.strip()][:5]
        script_parts.append(f"需給マトリックスを確認しましょう。{' '.join(lines)}")

    if "4_値嵩株影響（N5寄与率）" in sections:
        content = sections["4_値嵩株影響（N5寄与率）"]
        lines = [l.strip() for l in content.split("\n") if l.strip() and l.startswith("-")][:2]
        script_parts.append(f"値嵩株の影響度を見ていきます。{' '.join(lines)}")

    if "5_オプション市場" in sections:
        content = sections["5_オプション市場"]
        lines = [l.strip() for l in content.split("\n") if l.strip() and l.startswith("-")][:5]
        script_parts.append(f"オプション市場の状況です。{' '.join(lines)}")

    if "6_外国人投資家動向" in sections:
        content = sections["6_外国人投資家動向"]
        lines = [l.strip() for l in content.split("\n") if l.strip() and l.startswith("-")][:4]
        script_parts.append(f"外国人投資家の動向を確認します。{' '.join(lines)}")

    if "7_グローバル相関" in sections:
        content = sections["7_グローバル相関"]
        lines = [l.strip() for l in content.split("\n") if l.strip() and l.startswith("- 日経")]
        script_parts.append(f"グローバル市場との相関を見ていきましょう。{' '.join(lines)}")

    if "8_本日の重要ニュース" in sections:
        content = sections["8_本日の重要ニュース"]
        news_items = re.findall(r"(\d+)\.\s*(.+?)(?:\n|$)", content)
        if news_items:
            script_parts.append("本日の重要ニュースをお伝えします。")
            for num, headline in news_items[:3]:
                script_parts.append(f"{num}件目。{headline}")

    if "9_臥龍総括" in sections:
        content = sections["9_臥龍総括"]
        quote_match = re.search(r"諸葛亮曰く[―ー]\s*(.+?)(?:ただし、これは|━|$)", content, re.DOTALL)
        if quote_match:
            quote = quote_match.group(1).strip()
            quote = re.sub(r"[一二三四五六七八九十]に、", "。", quote)
            script_parts.append(f"最後に、臥龍からの総括です。{quote}")

    script_parts.append("以上、八門遁甲ナイトレポートでした。本レポートは情報提供を目的としており、投資助言ではありません。投資判断は自己責任でお願いいたします。")
    return "\n\n".join(script_parts)


def newsletter_to_script_morning(parsed: dict) -> str:
    """朝版ニュースレターを一人読み原稿に変換"""
    sections = parsed["sections"]
    date_str = parsed["date"]

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        date_readable = f"{dt.month}月{dt.day}日、{weekdays[dt.weekday()]}曜日"
    except:
        date_readable = date_str

    script_parts = [f"おはようございます。八門遁甲モーニングブリーフをお届けします。本日は{date_readable}です。"]

    for section_key, content in sections.items():
        if section_key == "header":
            continue
        section_name = section_key.split("_", 1)[-1] if "_" in section_key else section_key
        lines = [l.strip() for l in content.split("\n") if l.strip()][:5]
        script_parts.append(f"{section_name}です。{' '.join(lines)}")

    script_parts.append("以上、八門遁甲モーニングブリーフでした。本日も良いトレードを。")
    return "\n\n".join(script_parts)


def newsletter_to_script_lunch(parsed: dict) -> str:
    """昼版ニュースレターを一人読み原稿に変換"""
    content = parsed["raw_content"]
    date_str = parsed["date"]

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        date_readable = f"{dt.month}月{dt.day}日、{weekdays[dt.weekday()]}曜日"
    except:
        date_readable = date_str

    script_parts = [f"こんにちは。八門遁甲ランチレポートをお届けします。本日は{date_readable}です。本日の活況銘柄を分析していきます。"]

    stock_matches = re.findall(r"【(\d+)位】\s*(\d+)\s+(.+?)（([+-]?\d+\.?\d*)%）", content)
    for rank, code, name, change in stock_matches[:5]:
        script_parts.append(f"{rank}位は{name}、前日比{change}パーセントです。")

    mashoku_match = re.search(r"馬謖の昼餉コメント[━━]*\n(.+?)(?:====|$)", content, re.DOTALL)
    if mashoku_match:
        comment = mashoku_match.group(1).strip()[:200]
        script_parts.append(f"馬謖からのコメントです。{comment}")

    script_parts.append("以上、八門遁甲ランチレポートでした。後場も良いトレードを。")
    return "\n\n".join(script_parts)


def newsletter_to_script(parsed: dict) -> str:
    """ニュースレターを一人読み原稿に変換"""
    version = parsed["version"]
    if version == "night":
        return newsletter_to_script_night(parsed)
    elif version == "morning":
        return newsletter_to_script_morning(parsed)
    elif version == "lunch":
        return newsletter_to_script_lunch(parsed)
    return parsed["raw_content"]


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

        audio_segments.append(AudioSegment.silent(duration=pause_ms))

    combined = AudioSegment.empty()
    for seg in audio_segments:
        combined += seg

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="mp3", bitrate="192k")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate audio from newsletter')
    parser.add_argument('input', nargs='+', help='Input newsletter TXT file path(s)')
    parser.add_argument('--output-dir', '-o', type=str, default='output/newsletter',
                        help='Output directory for MP3 files')
    args = parser.parse_args()

    # VOICEVOX接続確認
    try:
        version_resp = requests.get(f"{VOICEVOX_URL}/version", timeout=5)
        print(f"VOICEVOX version: {version_resp.text}")
    except Exception as e:
        print(f"ERROR: VOICEVOX connection failed: {e}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    for input_file in args.input:
        txt_path = Path(input_file)
        if not txt_path.exists():
            print(f"ERROR: File not found: {txt_path}")
            continue

        print(f"\n=== Processing: {txt_path.name} ===")
        parsed = parse_newsletter_txt(txt_path)
        print(f"  Version: {parsed['version']}, Date: {parsed['date']}")

        script = newsletter_to_script(parsed)
        print(f"  Script length: {len(script)} chars")

        output_path = output_dir / f"{parsed['version']}_{parsed['date']}.mp3"
        print(f"  Generating audio...")
        generate_audio_from_script(script, output_path)

        file_size = output_path.stat().st_size
        print(f"  Output: {output_path} ({file_size:,} bytes)")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
