# 8-mon-assets

8-mon.com の静的アセット生成・配信用リポジトリ（Public）

## 概要

このリポジトリは [8-mon](https://8-mon.com) の以下のアセットを生成します：

- **臥龍ラジオ音声** (`/audio/*.mp3`)
- **スマイルカーブデータ** (`/smile-curve.json`)

生成されたファイルは **Cloudflare R2** (`r2.8-mon.com`) にアップロードされ、フロントエンドから参照されます。

## アーキテクチャ

```
[8-mon-assets (Public)]          [Cloudflare R2]          [8-mon (Private)]
     |                                |                         |
     |-- generate-radio.yml --------->|                         |
     |   (1時間ごと)                 audio/*.mp3                |
     |                                |                         |
     |-- fetch-smile-curve.yml ------>|                         |
     |   (1時間ごと)              smile-curve.json               |
     |                                |                         |
     |                                +<--- fetch ---- Frontend |
     |                                       (r2.8-mon.com)     |
```

## ディレクトリ構成

```
8-mon-assets/
├── radio/
│   ├── generate_radio_audio.py   # 音声生成スクリプト
│   └── requirements.txt
├── market-data/
│   ├── fetch_smile_curve.py      # スマイルカーブ取得
│   └── requirements.txt
├── scripts/
│   ├── upload_to_r2.py           # R2アップロード共通
│   └── requirements.txt
└── .github/workflows/
    ├── generate-radio.yml        # 音声生成ワークフロー
    └── fetch-smile-curve.yml     # スマイルカーブ取得
```

## 必要なSecrets

GitHub Repository Secrets に以下を設定：

| Secret | 説明 |
|--------|------|
| `R2_ENDPOINT` | Cloudflare R2 エンドポイント |
| `R2_ACCESS_KEY_ID` | R2 API Access Key ID |
| `R2_SECRET_ACCESS_KEY` | R2 API Secret Access Key |
| `R2_BUCKET` | R2 バケット名 (`8-mon-assets`) |

## ライセンス

### 音声モデル
- **つくよみちゃん Style-BERT-VITS2**
  - モデル提供: [ayousanz](https://huggingface.co/ayousanz/tsukuyomi-chan-style-bert-vits2-model)
  - キャラクター: [つくよみちゃん](https://tyc.rei-yumesaki.net/)
  - ライセンス: つくよみちゃんキャラクターライセンス
