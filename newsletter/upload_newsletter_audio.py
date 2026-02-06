#!/usr/bin/env python3
"""
ニュースレター音声ファイルをR2にアップロード

Usage:
    python upload_newsletter_audio.py output/newsletter/*.mp3

アップロード先:
    r2.8-mon.com/audio/newsletter/
    ├── night_2025-12-23.mp3
    ├── morning_2025-12-23.mp3
    ├── lunch_2025-12-23.mp3
    └── manifest.json
"""
import boto3
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: upload_newsletter_audio.py <mp3_files...>")
        sys.exit(1)

    # R2 credentials from environment
    endpoint_url = os.environ.get('R2_ENDPOINT')
    access_key = os.environ.get('R2_ACCESS_KEY_ID')
    secret_key = os.environ.get('R2_SECRET_ACCESS_KEY')
    bucket = os.environ.get('R2_BUCKET', '8-mon-assets')

    if not all([endpoint_url, access_key, secret_key]):
        print("Error: Missing R2 credentials")
        print("Required: R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")
        sys.exit(1)

    # Create S3 client for R2
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    # アップロード先プレフィックス
    prefix = 'audio/newsletter'

    # 現在のmanifestを取得（存在すれば）
    manifest = {"latest": {}, "generated_at": ""}
    try:
        response = s3.get_object(Bucket=bucket, Key=f'{prefix}/manifest.json')
        manifest = json.loads(response['Body'].read().decode('utf-8'))
        print(f"Existing manifest loaded: {manifest}")
    except s3.exceptions.NoSuchKey:
        print("No existing manifest, creating new one")
    except Exception as e:
        print(f"Warning: Could not load manifest: {e}")

    # MP3ファイルをアップロード
    uploaded_files = []
    for mp3_path in sys.argv[1:]:
        mp3 = Path(mp3_path)
        if not mp3.exists():
            print(f"Warning: File not found: {mp3}")
            continue

        if not mp3.suffix.lower() == '.mp3':
            print(f"Skipping non-MP3 file: {mp3}")
            continue

        filename = mp3.name
        r2_key = f'{prefix}/{filename}'

        print(f'Uploading {filename} to {r2_key}...')
        s3.upload_file(
            str(mp3),
            bucket,
            r2_key,
            ExtraArgs={
                'ContentType': 'audio/mpeg',
                'CacheControl': 'public, max-age=3600'
            }
        )
        uploaded_files.append(filename)

        # manifestを更新（版ごと）
        # ファイル名形式: night_2025-12-23.mp3, morning_2025-12-23.mp3, lunch_2025-12-23.mp3
        for version in ['night', 'morning', 'lunch']:
            if filename.startswith(f'{version}_'):
                manifest['latest'][version] = filename
                print(f"  Updated manifest.latest.{version} = {filename}")

    # manifestを更新してアップロード
    if uploaded_files:
        manifest['generated_at'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

        manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
        print(f"\nUpdated manifest:\n{manifest_json}")

        s3.put_object(
            Bucket=bucket,
            Key=f'{prefix}/manifest.json',
            Body=manifest_json.encode('utf-8'),
            ContentType='application/json',
            CacheControl='no-cache'
        )
        print(f"\nManifest uploaded to {prefix}/manifest.json")

    print(f"\n=== Upload complete ===")
    print(f"Uploaded {len(uploaded_files)} files: {uploaded_files}")


if __name__ == '__main__':
    main()
