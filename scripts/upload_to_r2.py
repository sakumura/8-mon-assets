#!/usr/bin/env python3
"""
Upload files to Cloudflare R2

Usage:
    python upload_to_r2.py audio ./output/audio
    python upload_to_r2.py data ./output/smile-curve.json
"""
import boto3
import os
import sys
from pathlib import Path


def get_content_type(filename: str) -> str:
    """Get content type based on file extension"""
    ext = Path(filename).suffix.lower()
    content_types = {
        '.mp3': 'audio/mpeg',
        '.json': 'application/json',
        '.wav': 'audio/wav',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
    }
    return content_types.get(ext, 'application/octet-stream')


def main():
    if len(sys.argv) < 3:
        print("Usage: upload_to_r2.py <type> <source_path>")
        print("  type: 'audio' or 'data'")
        print("  source_path: path to file or directory")
        sys.exit(1)

    upload_type = sys.argv[1]
    source_path = Path(sys.argv[2])

    # R2 credentials from environment
    endpoint_url = os.environ.get('R2_ENDPOINT')
    access_key = os.environ.get('R2_ACCESS_KEY_ID')
    secret_key = os.environ.get('R2_SECRET_ACCESS_KEY')
    bucket = os.environ.get('R2_BUCKET', '8-mon-assets')

    if not all([endpoint_url, access_key, secret_key]):
        print("Error: Missing R2 credentials")
        print("Required environment variables: R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")
        sys.exit(1)

    # Create S3 client for R2
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    if upload_type == 'audio':
        if not source_path.is_dir():
            print(f"Error: {source_path} is not a directory")
            sys.exit(1)

        # Upload all MP3 files
        for mp3 in source_path.glob('*.mp3'):
            print(f'Uploading {mp3.name}...')
            s3.upload_file(
                str(mp3),
                bucket,
                f'audio/{mp3.name}',
                ExtraArgs={'ContentType': 'audio/mpeg'}
            )

        # Upload manifest if exists
        manifest = source_path / 'audio-manifest.json'
        if manifest.exists():
            print('Uploading audio-manifest.json...')
            s3.upload_file(
                str(manifest),
                bucket,
                'audio/audio-manifest.json',
                ExtraArgs={'ContentType': 'application/json'}
            )

    elif upload_type == 'data':
        if not source_path.is_file():
            print(f"Error: {source_path} is not a file")
            sys.exit(1)

        filename = source_path.name
        content_type = get_content_type(filename)

        print(f'Uploading {filename}...')
        s3.upload_file(
            str(source_path),
            bucket,
            filename,
            ExtraArgs={'ContentType': content_type}
        )

    else:
        print(f"Error: Unknown upload type '{upload_type}'")
        print("Valid types: 'audio', 'data'")
        sys.exit(1)

    print('Upload complete!')


if __name__ == '__main__':
    main()
