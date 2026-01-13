"""
AI臥龍日記 音声アップロードスクリプト
生成した音声ファイルをCloudflare R2にアップロード
"""
import os
import sys
import json
import argparse
from pathlib import Path

import boto3
from botocore.config import Config

# R2設定（環境変数から取得）
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET", "8-mon-assets")


def get_r2_client():
    """R2クライアントを作成"""
    if not all([R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        raise ValueError("R2 credentials not set. Required: R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )


def upload_file(client, local_path: Path, r2_key: str, content_type: str = "audio/mpeg"):
    """ファイルをR2にアップロード"""
    print(f"  Uploading: {local_path.name} -> {r2_key}")

    with open(local_path, "rb") as f:
        client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=r2_key,
            Body=f,
            ContentType=content_type,
        )

    print(f"  Uploaded: https://r2.8-mon.com/{r2_key}")


def main():
    parser = argparse.ArgumentParser(description='Upload diary audio to R2')
    parser.add_argument('--input-dir', '-i', type=str, default='output/diary',
                        help='Input directory containing MP3 files')
    parser.add_argument('--date', '-d', type=str, help='Only upload files for specific date')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    # アップロード対象ファイル
    if args.date:
        mp3_files = list(input_dir.glob(f"*_{args.date}.mp3"))
    else:
        mp3_files = list(input_dir.glob("*.mp3"))

    if not mp3_files:
        print("No MP3 files found to upload")
        return

    print(f"\n=== Uploading {len(mp3_files)} file(s) to R2 ===")

    if args.dry_run:
        for mp3_file in mp3_files:
            r2_key = f"audio/diary/{mp3_file.name}"
            print(f"  [DRY-RUN] {mp3_file.name} -> {r2_key}")
        return

    # R2クライアント
    try:
        client = get_r2_client()
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # MP3ファイルをアップロード
    for mp3_file in mp3_files:
        r2_key = f"audio/diary/{mp3_file.name}"
        upload_file(client, mp3_file, r2_key)

    # マニフェストもアップロード
    manifest_path = input_dir / "manifest.json"
    if manifest_path.exists():
        upload_file(client, manifest_path, "audio/diary/manifest.json", content_type="application/json")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
