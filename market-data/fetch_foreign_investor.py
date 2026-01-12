"""
財務省「対外及び対内証券売買契約等の状況」データを取得・分析

データソース: https://www.mof.go.jp/policy/international_policy/reference/itn_transactions_in_securities/data.htm
- week.csv: 週次データ
- montha1.csv: 月次データ

対象: 対内証券投資（Portfolio Investment Liabilities）> 株式・投資ファンド持分のNet値
"""

import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from io import StringIO
import requests

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# Configuration
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

MOF_BASE_URL = "https://www.mof.go.jp/policy/international_policy/reference/itn_transactions_in_securities"
WEEK_CSV_URL = f"{MOF_BASE_URL}/week.csv"
MONTH_CSV_URL = f"{MOF_BASE_URL}/montha1.csv"

# 日経225のヒストリカルデータ（リード・ラグ分析用）
NIKKEI_DATA_PATH = Path(__file__).parent.parent.parent / "8-mon" / "frontend" / "public" / "data.json"


def normalize_fullwidth(text: str) -> str:
    """全角文字を半角に正規化"""
    # 全角→半角の変換テーブル
    replacements = {
        '．': '.',  # U+FF0E → .
        '～': '~',  # U+FF5E → ~
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
        '，': ',',  # 全角カンマ
        '　': ' ',  # 全角スペース
    }
    for full, half in replacements.items():
        text = text.replace(full, half)
    return text


def fetch_csv(url: str) -> pd.DataFrame | None:
    """CSVファイルをダウンロードしてDataFrameに変換"""
    logging.info(f"Fetching CSV from: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # 複数のエンコーディングを試行
        content = None
        for encoding in ['utf-8', 'shift_jis', 'cp932', 'iso-8859-1', 'latin-1']:
            try:
                content = response.content.decode(encoding)
                logging.info(f"Successfully decoded with {encoding}")
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            # 最後の手段: errors='ignore' で強制デコード
            content = response.content.decode('utf-8', errors='ignore')
            logging.warning("Decoded with utf-8 (ignoring errors)")

        # 全角文字を半角に正規化
        content = normalize_fullwidth(content)

        # CSVをパース（ヘッダー行が複数ある特殊フォーマット対応）
        df = pd.read_csv(StringIO(content), header=None)

        return df

    except Exception as e:
        logging.error(f"Failed to fetch CSV: {e}")
        return None


def parse_weekly_data(df: pd.DataFrame) -> list[dict]:
    """
    週次CSVをパースして株式フローデータを抽出

    CSVフォーマット（財務省週次データ）:
    - Row 8: "2. Portfolio Investment Liabilities" が col 12 にある
    - Row 11: "Equity and investment fund shares" が col 12 にある
    - Row 13: "Net" が col 14 にある（対内証券投資の株式Net）
    - Row 14以降: データ行
    - Col 0: 日付（例: "2005.1.2～ 1.8"）
    - Col 14: 対内証券投資 > 株式・投資ファンド持分 > Net
    """
    logging.info("Parsing weekly data...")

    data = []
    import re

    # CSVの構造を確認
    # 対内証券投資（Portfolio Investment Liabilities）の株式Net列を特定
    equity_net_col = None
    data_start_row = None

    # ヘッダー行を探す
    for idx in range(min(20, len(df))):
        for col_idx in range(len(df.columns)):
            val = str(df.iloc[idx, col_idx]) if pd.notna(df.iloc[idx, col_idx]) else ''
            if 'Portfolio Investment Liabilities' in val or '対内証券投資' in val:
                # このセクションの開始位置を記録
                liabilities_start_col = col_idx
                logging.info(f"Found Liabilities section at row {idx}, col {col_idx}")

                # このセクションの株式Net列を探す（通常col+2）
                # Row 13に "Net" がある列を探す
                for net_row in range(idx, min(idx + 10, len(df))):
                    for net_col in range(liabilities_start_col, min(liabilities_start_col + 10, len(df.columns))):
                        net_val = str(df.iloc[net_row, net_col]) if pd.notna(df.iloc[net_row, net_col]) else ''
                        if net_val.strip() == 'Net':
                            equity_net_col = net_col
                            data_start_row = net_row + 1
                            logging.info(f"Found equity Net column at {net_col}, data starts at row {data_start_row}")
                            break
                    if equity_net_col:
                        break
                break
        if equity_net_col:
            break

    # フォールバック: 固定位置を使用
    if equity_net_col is None:
        logging.info("Using fixed column position for equity net (col 14)")
        equity_net_col = 14
        data_start_row = 14

    # 日付列は0番目
    date_col = 0

    # データ行を処理
    for idx in range(data_start_row, len(df)):
        try:
            date_val = df.iloc[idx, date_col]
            net_val = df.iloc[idx, equity_net_col]

            # 日付のパース
            if pd.isna(date_val):
                continue

            date_str = str(date_val).strip()

            # 日付フォーマットの判定（財務省形式: "2005.1.2～ 1.8"）
            parsed_date = None

            # パターン1: "2005.1.2～ 1.8" または "2005.12.25～12.31"
            match = re.match(r'(\d{4})\.(\d{1,2})\.(\d{1,2}).*?(\d{1,2})\.(\d{1,2})', date_str)
            if match:
                year, month1, day1, month2, day2 = match.groups()
                # 週の終わりの日付を使用
                parsed_date = datetime(int(year), int(month2), int(day2))
            else:
                # パターン2: "2005.1.2～ 1.8" (月が省略)
                match = re.match(r'(\d{4})\.(\d{1,2})\.(\d{1,2}).*?(\d{1,2})', date_str)
                if match:
                    year, month, day1, day2 = match.groups()
                    parsed_date = datetime(int(year), int(month), int(day2))

            if parsed_date is None:
                # その他の標準形式を試行
                for fmt in ['%Y/%m/%d', '%Y-%m-%d']:
                    try:
                        parsed_date = datetime.strptime(date_str.split('～')[0].strip(), fmt)
                        break
                    except ValueError:
                        continue

            if parsed_date is None:
                continue

            # 数値のパース
            if pd.isna(net_val):
                continue

            net_str = str(net_val).replace(',', '').replace(' ', '').strip()
            if net_str in ['', '-', 'nan', 'NaN']:
                continue

            try:
                net_value = float(net_str)
            except ValueError:
                continue

            # 2010年以降のデータのみ使用
            if parsed_date.year < 2010:
                continue

            data.append({
                'date': parsed_date.strftime('%Y-%m-%d'),
                'net': net_value  # 億円単位（CSVの単位は100 million Yen = 億円）
            })

        except Exception as e:
            continue

    # 日付でソート
    data.sort(key=lambda x: x['date'])

    logging.info(f"Parsed {len(data)} weekly data points")
    return data


def parse_monthly_data(df: pd.DataFrame) -> list[dict]:
    """
    月次CSVをパースして株式フローデータを抽出

    CSVフォーマット（財務省月次データ）:
    - Row 11以降: データ行
    - Col 0: 年（2005など、または空欄）
    - Col 2: 月（Jan, Feb, Mar等）
    - Col 16: 対内証券投資 > 株式・投資ファンド持分 > Net
    """
    logging.info("Parsing monthly data...")

    data = []

    # 月名の変換テーブル
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    # データ開始行とNet列を特定
    equity_net_col = 16  # 月次CSVでは固定
    data_start_row = 11  # 月次CSVでは固定

    current_year = None

    for idx in range(data_start_row, len(df)):
        try:
            # 年の取得（空欄の場合は前の年を継続）
            year_val = df.iloc[idx, 0]
            if pd.notna(year_val):
                year_str = str(year_val).strip()
                # "2005" または "(平成17年) " などから年を抽出
                import re
                year_match = re.search(r'(\d{4})', year_str)
                if year_match:
                    current_year = int(year_match.group(1))

            if current_year is None:
                continue

            # 月の取得
            month_val = df.iloc[idx, 2]
            if pd.isna(month_val):
                continue

            month_str = str(month_val).strip()
            if month_str not in month_map:
                continue

            month = month_map[month_str]

            # Net値の取得
            net_val = df.iloc[idx, equity_net_col]
            if pd.isna(net_val):
                continue

            net_str = str(net_val).replace(',', '').replace(' ', '').strip()
            if net_str in ['', '-', 'nan', 'NaN']:
                continue

            try:
                net_value = float(net_str)
            except ValueError:
                continue

            # 2010年以降のデータのみ使用
            if current_year < 2010:
                continue

            # 月末日を算出
            if month == 12:
                next_month_start = datetime(current_year + 1, 1, 1)
            else:
                next_month_start = datetime(current_year, month + 1, 1)

            from datetime import timedelta
            month_end = next_month_start - timedelta(days=1)

            data.append({
                'date': month_end.strftime('%Y-%m-%d'),
                'net': net_value
            })

        except Exception as e:
            continue

    # 日付でソート
    data.sort(key=lambda x: x['date'])

    logging.info(f"Parsed {len(data)} monthly data points")
    return data


def load_nikkei_data() -> pd.DataFrame | None:
    """日経225データを読み込む（ローカル優先、R2フォールバック）"""
    try:
        # 1. ローカルファイルを試行
        paths_to_try = [
            NIKKEI_DATA_PATH,
            Path(__file__).parent.parent.parent / "8-mon" / "frontend" / "public" / "data.json",
            Path("C:/Users/sakum/OneDrive/Projects/8-mon/frontend/public/data.json"),
        ]

        for path in paths_to_try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                df = pd.DataFrame(data)
                df['Date'] = pd.to_datetime(df['Date'])
                logging.info(f"Loaded Nikkei data from {path}: {len(df)} records")
                return df

        # 2. R2からフェッチ（GitHub Actions環境用）
        logging.info("Local file not found, fetching from R2...")
        r2_url = "https://r2.8-mon.com/data.json"
        response = requests.get(r2_url, timeout=30)
        if response.ok:
            data = response.json()
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'])
            logging.info(f"Loaded Nikkei data from R2: {len(df)} records")
            return df

        logging.warning("Nikkei data not available from local or R2")
        return None

    except Exception as e:
        logging.error(f"Failed to load Nikkei data: {e}")
        return None


def calculate_lead_lag(weekly_data: list[dict], nikkei_df: pd.DataFrame | None, max_lag: int = 12) -> dict:
    """
    累積フローと日経225の将来リターン相関を計算

    累積フローは長期（6-8週後）のリターンと最も強い相関を示す傾向がある。
    週次ネットフローは短期（3-4週後）でピークとなる。

    Args:
        weekly_data: 週次フローデータ（cumulative含む）
        nikkei_df: 日経225日次データ
        max_lag: 最大先行週数

    Returns:
        リード・ラグ分析結果
    """
    logging.info("Calculating cumulative flow vs future return correlation...")

    result = {
        'peak_lag': 0,
        'peak_correlation': 0.0,
        'interpretation': '',
        'correlations': []
    }

    if not weekly_data or nikkei_df is None or len(nikkei_df) == 0:
        result['interpretation'] = 'データ不足のため分析不可'
        return result

    try:
        # 週次フローをDataFrameに変換
        flow_df = pd.DataFrame(weekly_data)
        flow_df['date'] = pd.to_datetime(flow_df['date'])
        flow_df = flow_df.sort_values('date')

        logging.info(f"Flow data range: {flow_df['date'].min()} to {flow_df['date'].max()}")

        # 日経データの準備
        nikkei_df = nikkei_df.copy()
        nikkei_df['Date'] = pd.to_datetime(nikkei_df['Date'])
        logging.info(f"Nikkei data range: {nikkei_df['Date'].min()} to {nikkei_df['Date'].max()}")

        # 日経225を週次にリサンプル（土曜日基準 - MOFデータに合わせる）
        nikkei_df = nikkei_df.set_index('Date').sort_index()
        nikkei_weekly = nikkei_df['Nikkei_Close'].resample('W-SAT').last().dropna()

        logging.info(f"Nikkei weekly resampled: {len(nikkei_weekly)} weeks")

        # 累積フローと日経を結合
        flow_df = flow_df.set_index('date')
        combined = pd.DataFrame({
            'cumulative': flow_df['cumulative'],
            'net': flow_df['net'],
            'nikkei': nikkei_weekly
        }).dropna()

        logging.info(f"Combined data points: {len(combined)}")

        if len(combined) < 20:
            result['interpretation'] = f'データポイント不足（{len(combined)}週、20週以上必要）'
            return result

        # 直近52週に限定
        combined = combined.tail(52)
        logging.info(f"Using last {len(combined)} weeks for analysis")

        # 累積フロー vs 将来N週後リターンの相関を計算
        correlations = []
        weeks_to_check = [1, 2, 3, 4, 5, 6, 8, 12]

        for weeks_ahead in weeks_to_check:
            # N週後のリターンを計算
            future_return = combined['nikkei'].pct_change(weeks_ahead).shift(-weeks_ahead) * 100

            # 有効なデータのみで相関計算
            valid_idx = ~future_return.isna()
            if valid_idx.sum() > 10:
                corr = combined.loc[valid_idx, 'cumulative'].corr(future_return[valid_idx])
                if not np.isnan(corr):
                    correlations.append({
                        'lag': weeks_ahead,
                        'correlation': round(corr, 3)
                    })
                    logging.debug(f"{weeks_ahead}週後リターン: r = {corr:.3f}")

        result['correlations'] = correlations

        # ピーク相関を特定
        if correlations:
            peak = max(correlations, key=lambda x: x['correlation'])  # 正の相関が高いものを選択
            result['peak_lag'] = peak['lag']
            result['peak_correlation'] = peak['correlation']

            # 解釈テキストを生成
            if peak['correlation'] < 0.2:
                result['interpretation'] = f"弱い相関（r={peak['correlation']:.2f}）。累積フローと将来リターンの連動性は限定的。"
            elif peak['correlation'] < 0.35:
                result['interpretation'] = f"累積フローは{peak['lag']}週後リターンと中程度の相関（r={peak['correlation']:.2f}）。買い越し継続で上昇傾向。"
            else:
                result['interpretation'] = f"累積フローは{peak['lag']}週後リターンと強い相関（r={peak['correlation']:.2f}）。外国人買い越しは中期上昇の先行指標。"

        logging.info(f"Lead-lag analysis: peak at {result['peak_lag']} weeks ahead with correlation {result['peak_correlation']}")

    except Exception as e:
        logging.error(f"Lead-lag calculation failed: {e}")
        import traceback
        traceback.print_exc()
        result['interpretation'] = f'分析エラー: {str(e)}'

    return result


def calculate_statistics(data: list[dict]) -> dict:
    """統計量を計算"""
    if not data:
        return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}

    values = [d['net'] for d in data]
    return {
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values))
    }


def calculate_percentile(value: float, data: list[dict]) -> float:
    """パーセンタイルを計算"""
    if not data:
        return 50.0

    values = sorted([d['net'] for d in data])
    count_below = sum(1 for v in values if v < value)
    return round((count_below / len(values)) * 100, 1)


def calculate_cumulative(data: list[dict], weeks: int = 52) -> float:
    """累積値を計算"""
    recent = data[-weeks:] if len(data) >= weeks else data
    return sum(d['net'] for d in recent)


def calculate_monthly_heatmap(monthly_data: list[dict]) -> list[dict]:
    """月別の平均フローを計算（ヒートマップ用）"""
    if not monthly_data:
        return []

    # 月別に集計
    monthly_sums = {}
    monthly_counts = {}

    for d in monthly_data:
        month = int(d['date'][5:7])
        if month not in monthly_sums:
            monthly_sums[month] = 0
            monthly_counts[month] = 0
        monthly_sums[month] += d['net']
        monthly_counts[month] += 1

    heatmap = []
    for month in range(1, 13):
        if month in monthly_sums and monthly_counts[month] > 0:
            avg = monthly_sums[month] / monthly_counts[month]
            heatmap.append({
                'month': month,
                'avg_net': round(avg, 0),
                'is_buying': avg > 0
            })
        else:
            heatmap.append({
                'month': month,
                'avg_net': 0,
                'is_buying': None
            })

    return heatmap


def generate_output(weekly_data: list[dict], monthly_data: list[dict], lead_lag: dict) -> dict:
    """出力JSONを生成"""

    # 直近52週のデータ
    recent_weekly = weekly_data[-52:] if len(weekly_data) >= 52 else weekly_data

    # 累積計算
    cumulative = 0
    for d in recent_weekly:
        cumulative += d['net']
        d['cumulative'] = cumulative

    latest = recent_weekly[-1] if recent_weekly else None

    output = {
        'generated_at': datetime.now().isoformat(),
        'weekly': {
            'data': recent_weekly,
            'latest': {
                'date': latest['date'] if latest else None,
                'net': latest['net'] if latest else 0,
                'net_billion': round(latest['net'], 0) if latest else 0,  # 億円単位（netは既に億円）
                'percentile': calculate_percentile(latest['net'], recent_weekly) if latest else 50
            },
            'cumulative_52w': calculate_cumulative(weekly_data, 52),
            'stats': calculate_statistics(recent_weekly)
        },
        'monthly': {
            'data': monthly_data[-24:] if len(monthly_data) >= 24 else monthly_data,  # 直近24ヶ月
            'heatmap': calculate_monthly_heatmap(monthly_data)
        },
        'lead_lag': lead_lag
    }

    return output


def save_output(data: dict, output_path: Path):
    """JSONファイルに保存"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logging.info(f"Output saved to: {output_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='Fetch foreign investor flow data from MOF')
    parser.add_argument('--output-dir', type=str, help='Output directory path')
    parser.add_argument('--skip-lead-lag', action='store_true', help='Skip lead-lag analysis')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_path = output_dir / "foreign-investor.json"

    # 週次データ取得
    week_df = fetch_csv(WEEK_CSV_URL)
    if week_df is None:
        logging.error("Failed to fetch weekly data")
        return 1

    weekly_data = parse_weekly_data(week_df)

    # 累積フローを事前計算（lead-lag分析に必要）
    recent_weekly = weekly_data[-52:] if len(weekly_data) >= 52 else weekly_data
    cumulative = 0
    for d in recent_weekly:
        cumulative += d['net']
        d['cumulative'] = cumulative

    # 月次データ取得
    month_df = fetch_csv(MONTH_CSV_URL)
    monthly_data = parse_monthly_data(month_df) if month_df is not None else []

    # リード・ラグ分析（累積フローベース）
    lead_lag = {'peak_lag': 0, 'peak_correlation': 0, 'interpretation': 'スキップ', 'correlations': []}
    if not args.skip_lead_lag:
        nikkei_df = load_nikkei_data()
        lead_lag = calculate_lead_lag(recent_weekly, nikkei_df)

    # 出力生成
    output = generate_output(weekly_data, monthly_data, lead_lag)

    # 保存
    save_output(output, output_path)

    # サマリー表示
    print(f"\n=== Foreign Investor Flow Data ===")
    print(f"Generated: {output['generated_at']}")
    print(f"Weekly data points: {len(output['weekly']['data'])}")
    if output['weekly']['latest']['date']:
        print(f"Latest: {output['weekly']['latest']['date']} - {output['weekly']['latest']['net_billion']:.0f}億円 (P{output['weekly']['latest']['percentile']:.0f})")
    print(f"52-week cumulative: {output['weekly']['cumulative_52w']:.0f}億円")
    print(f"Lead-lag: {output['lead_lag']['interpretation']}")

    return 0


if __name__ == '__main__':
    exit(main())
