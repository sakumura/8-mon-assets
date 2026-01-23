"""
財務省「対外及び対内証券売買契約等の状況」データを取得・分析

データソース: https://www.mof.go.jp/policy/international_policy/reference/itn_transactions_in_securities/data.htm
- week.csv: 週次データ
- montha1.csv: 月次データ

対象:
- 株式: 対内証券投資（Portfolio Investment Liabilities）> 株式・投資ファンド持分のNet値
- 中長期債: 対内証券投資（Portfolio Investment Liabilities）> 中長期債のNet値

追加データ:
- 10年国債利回り: FRED API（IRLTLT01JPM156N）
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

# FRED API for 10-year JGB yield
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_JGB_SERIES_ID = "IRLTLT01JPM156N"  # Long-Term Government Bond Yields: 10-year: Main (Including Benchmark) for Japan

# 日経225のヒストリカルデータ（リード・ラグ分析用）
NIKKEI_DATA_PATH = Path(__file__).parent.parent.parent / "8-mon" / "frontend" / "public" / "data.json"


import re

def normalize_fullwidth(text: str) -> str:
    """全角文字を半角に正規化"""
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


def parse_mof_weekly_date(raw_value, min_year: int = 2010) -> datetime | None:
    """
    財務省週次データの日付をパース

    CSVフォーマット:
    - "2005.1.2～ 1.8" (週の開始～終了)
    - "2005.12.25～12.31" (月を含む場合)

    Args:
        raw_value: DataFrameセルの値
        min_year: この年より古いデータは None を返す

    Returns:
        週の終わりの日付、またはパース失敗/フィルタリング時は None
    """
    if pd.isna(raw_value):
        return None

    date_str = normalize_fullwidth(str(raw_value).strip())
    parsed_date = None

    # パターン1: "2005.1.2～ 1.8" または "2005.12.25～12.31"
    # ～の後の部分を分離してパース
    if '~' in date_str:
        parts = date_str.split('~')
        start_part = parts[0].strip()
        end_part = parts[1].strip() if len(parts) > 1 else ''

        # 開始日から年を取得
        start_match = re.match(r'(\d{4})\.(\d{1,2})\.(\d{1,2})', start_part)
        if start_match:
            year = int(start_match.group(1))
            start_month = int(start_match.group(2))

            # 終了日をパース
            end_match = re.match(r'(\d{1,2})\.(\d{1,2})', end_part)
            if end_match:
                end_month = int(end_match.group(1))
                end_day = int(end_match.group(2))
                parsed_date = datetime(year, end_month, end_day)
            else:
                # 日のみの場合 (例: "1.8" -> 日=8、月は開始月と同じ)
                day_match = re.match(r'(\d{1,2})', end_part)
                if day_match:
                    end_day = int(day_match.group(1))
                    parsed_date = datetime(year, start_month, end_day)
    else:
        # パターン2: "2005.1.2～ 1.8" (月が省略)
        match = re.match(r'(\d{4})\.(\d{1,2})\.(\d{1,2}).*?(\d{1,2})', date_str)
        if match:
            year, month, day1, day2 = match.groups()
            parsed_date = datetime(int(year), int(month), int(day2))

    # フォールバック: 標準フォーマット
    if parsed_date is None:
        for fmt in ['%Y/%m/%d', '%Y-%m-%d']:
            try:
                parsed_date = datetime.strptime(date_str.split('～')[0].strip(), fmt)
                break
            except ValueError:
                continue

    # 年フィルタリング
    if parsed_date and parsed_date.year < min_year:
        return None

    return parsed_date


def parse_net_value(raw_value) -> float | None:
    """
    Net値をパース

    Args:
        raw_value: DataFrameセルの値

    Returns:
        パースされた数値、または None
    """
    if pd.isna(raw_value):
        return None

    net_str = str(raw_value).replace(',', '').replace(' ', '').strip()
    if net_str in ['', '-', 'nan', 'NaN']:
        return None

    try:
        return float(net_str)
    except ValueError:
        return None


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


def find_net_column(
    df: pd.DataFrame,
    section_keywords: list[str],
    subsection_keywords: list[str] | None = None,
    fallback_col: int | None = None
) -> tuple[int | None, int | None]:
    """
    財務省CSVから特定セクションのNet列を検出

    Args:
        df: CSVデータフレーム
        section_keywords: メインセクションのキーワードリスト
        subsection_keywords: サブセクションのキーワードリスト（Noneの場合は直接Net列を探す）
        fallback_col: 検出失敗時のフォールバック列番号

    Returns:
        (net_column, data_start_row) のタプル
    """
    for idx in range(min(20, len(df))):
        for col_idx in range(len(df.columns)):
            val = str(df.iloc[idx, col_idx]) if pd.notna(df.iloc[idx, col_idx]) else ''

            if any(keyword in val for keyword in section_keywords):
                section_col = col_idx
                logging.info(f"Found section at row {idx}, col {col_idx}")

                # サブセクションが指定されている場合
                if subsection_keywords:
                    for search_row in range(idx, min(idx + 15, len(df))):
                        for search_col in range(section_col, min(section_col + 20, len(df.columns))):
                            search_val = str(df.iloc[search_row, search_col]) if pd.notna(df.iloc[search_row, search_col]) else ''

                            if any(keyword in search_val for keyword in subsection_keywords):
                                logging.info(f"Found subsection at row {search_row}, col {search_col}")
                                # Net列を探す
                                for net_row in range(search_row, min(search_row + 5, len(df))):
                                    for net_col in range(search_col, min(search_col + 5, len(df.columns))):
                                        net_val = str(df.iloc[net_row, net_col]) if pd.notna(df.iloc[net_row, net_col]) else ''
                                        if net_val.strip() == 'Net':
                                            return net_col, net_row + 1
                else:
                    # サブセクションなし: 直接Net列を探す
                    for net_row in range(idx, min(idx + 10, len(df))):
                        for net_col in range(section_col, min(section_col + 10, len(df.columns))):
                            net_val = str(df.iloc[net_row, net_col]) if pd.notna(df.iloc[net_row, net_col]) else ''
                            if net_val.strip() == 'Net':
                                return net_col, net_row + 1

    # フォールバック
    if fallback_col is not None:
        logging.warning(f"Using fallback column {fallback_col}")
        return fallback_col, 14

    return None, None


def parse_weekly_flow_data(
    df: pd.DataFrame,
    net_col: int,
    data_start_row: int,
    label: str = "data"
) -> list[dict]:
    """
    週次CSVからフローデータを抽出（株式・債券共通）

    Args:
        df: CSVデータフレーム
        net_col: Net列の番号
        data_start_row: データ開始行
        label: ログ用ラベル

    Returns:
        フローデータのリスト
    """
    data = []
    date_col = 0

    for idx in range(data_start_row, len(df)):
        parsed_date = parse_mof_weekly_date(df.iloc[idx, date_col])
        if parsed_date is None:
            continue

        net_value = parse_net_value(df.iloc[idx, net_col])
        if net_value is None:
            continue

        data.append({
            'date': parsed_date.strftime('%Y-%m-%d'),
            'net': net_value  # 億円単位
        })

    data.sort(key=lambda x: x['date'])
    logging.info(f"Parsed {len(data)} weekly {label} data points")
    return data


def parse_weekly_data(df: pd.DataFrame) -> list[dict]:
    """
    週次CSVをパースして株式フローデータを抽出

    CSVフォーマット（財務省週次データ）:
    - 対内証券投資 > 株式・投資ファンド持分 > Net
    """
    logging.info("Parsing weekly stock data...")

    net_col, data_start_row = find_net_column(
        df,
        section_keywords=['Portfolio Investment Liabilities', '対内証券投資'],
        subsection_keywords=None,  # 株式は最初のNet列
        fallback_col=14
    )

    if net_col is None:
        logging.error("Failed to find equity Net column")
        return []

    return parse_weekly_flow_data(df, net_col, data_start_row, label="stock")


def parse_weekly_bond_data(df: pd.DataFrame) -> list[dict]:
    """
    週次CSVをパースして中長期債フローデータを抽出

    CSVフォーマット（財務省週次データ）:
    - 対内証券投資 > 中長期債 > Net
    """
    logging.info("Parsing weekly bond data...")

    net_col, data_start_row = find_net_column(
        df,
        section_keywords=['Portfolio Investment Liabilities', '対内証券投資'],
        subsection_keywords=['Long-term debt securities', '中長期債'],
        fallback_col=9
    )

    if net_col is None:
        logging.error("Failed to find bond Net column")
        return []

    return parse_weekly_flow_data(df, net_col, data_start_row, label="bond")


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


def fetch_jgb_yield() -> pd.DataFrame | None:
    """
    FRED APIから10年国債利回りを取得

    FRED Series: IRLTLT01JPM156N
    - Long-Term Government Bond Yields: 10-year: Main for Japan
    - 月次データ（%単位）
    """
    logging.info("Fetching 10-year JGB yield from FRED...")

    try:
        # FREDから直接CSVを取得（APIキー不要）
        csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_JGB_SERIES_ID}&cosd=2010-01-01"

        response = requests.get(csv_url, timeout=30)
        if not response.ok:
            logging.warning(f"Failed to fetch JGB yield: {response.status_code}")
            return None

        # CSVをパース
        content = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(content))

        # カラム名を正規化
        df.columns = ['date', 'yield']
        df['date'] = pd.to_datetime(df['date'])
        df['yield'] = pd.to_numeric(df['yield'], errors='coerce')
        df = df.dropna()

        logging.info(f"Fetched {len(df)} JGB yield data points")
        return df

    except Exception as e:
        logging.error(f"Failed to fetch JGB yield: {e}")
        return None


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


def calculate_bond_yield_correlation(weekly_bond_data: list[dict], jgb_yield_df: pd.DataFrame | None) -> dict:
    """
    債券フローと10年国債利回りの相関を計算

    債券売り越し（負のフロー）→ 金利上昇圧力
    債券買い越し（正のフロー）→ 金利低下圧力

    Args:
        weekly_bond_data: 週次債券フローデータ（cumulative含む）
        jgb_yield_df: 10年国債利回りデータ

    Returns:
        相関分析結果
    """
    logging.info("Calculating bond flow vs JGB yield correlation...")

    result = {
        'correlation': 0.0,
        'yield_latest': None,
        'interpretation': ''
    }

    if not weekly_bond_data or jgb_yield_df is None or len(jgb_yield_df) == 0:
        result['interpretation'] = 'データ不足のため分析不可'
        return result

    try:
        # 週次フローをDataFrameに変換
        flow_df = pd.DataFrame(weekly_bond_data)
        flow_df['date'] = pd.to_datetime(flow_df['date'])
        flow_df = flow_df.sort_values('date').set_index('date')

        # 金利データを週次にリサンプル（月次データなので、ffillで補間）
        jgb_yield_df = jgb_yield_df.copy()
        jgb_yield_df = jgb_yield_df.set_index('date').sort_index()
        jgb_weekly = jgb_yield_df['yield'].resample('W-SAT').ffill().dropna()

        # 結合
        combined = pd.DataFrame({
            'cumulative': flow_df['cumulative'] if 'cumulative' in flow_df.columns else flow_df['net'].cumsum(),
            'net': flow_df['net'],
            'yield': jgb_weekly
        }).dropna()

        logging.info(f"Combined bond/yield data points: {len(combined)}")

        if len(combined) < 20:
            result['interpretation'] = f'データポイント不足（{len(combined)}週、20週以上必要）'
            return result

        # 直近52週に限定
        combined = combined.tail(52)

        # 累積フローと金利の相関（通常は負の相関: 売り越し→金利上昇）
        corr = combined['cumulative'].corr(combined['yield'])

        if not np.isnan(corr):
            result['correlation'] = round(corr, 3)

        # 直近の金利
        result['yield_latest'] = round(float(combined['yield'].iloc[-1]), 2)

        # 解釈テキスト
        if result['correlation'] < -0.3:
            result['interpretation'] = f"累積フローと金利は負の相関（r={result['correlation']:.2f}）。債券売越→金利上昇圧力。"
        elif result['correlation'] > 0.3:
            result['interpretation'] = f"累積フローと金利は正の相関（r={result['correlation']:.2f}）。通常と逆のパターン。"
        else:
            result['interpretation'] = f"累積フローと金利の相関は弱い（r={result['correlation']:.2f}）。"

        logging.info(f"Bond-yield correlation: {result['correlation']}, latest yield: {result['yield_latest']}%")

    except Exception as e:
        logging.error(f"Bond-yield correlation calculation failed: {e}")
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


def generate_output(
    weekly_stock_data: list[dict],
    monthly_stock_data: list[dict],
    stock_lead_lag: dict,
    weekly_bond_data: list[dict],
    monthly_bond_data: list[dict],
    bond_yield_correlation: dict
) -> dict:
    """出力JSONを生成（stock/bond構造）"""

    def build_section(weekly_data: list[dict], monthly_data: list[dict]) -> dict:
        """株式または債券セクションを構築（入力を変更しない）"""
        recent_weekly_slice = weekly_data[-52:] if len(weekly_data) >= 52 else weekly_data

        # コピーを作成して変更（副作用を防ぐ）
        recent_weekly = []
        cumulative = 0
        for d in recent_weekly_slice:
            new_d = d.copy()
            cumulative += d['net']
            new_d['cumulative'] = cumulative
            recent_weekly.append(new_d)

        latest = recent_weekly[-1] if recent_weekly else None

        return {
            'weekly': {
                'data': recent_weekly,
                'latest': {
                    'date': latest['date'] if latest else None,
                    'net': latest['net'] if latest else 0,
                    'net_billion': round(latest['net'], 0) if latest else 0,
                    'percentile': calculate_percentile(latest['net'], recent_weekly) if latest else 50
                },
                'cumulative_52w': calculate_cumulative(weekly_data, 52),
                'stats': calculate_statistics(recent_weekly)
            },
            'monthly': {
                'data': monthly_data[-24:] if len(monthly_data) >= 24 else monthly_data,
                'heatmap': calculate_monthly_heatmap(monthly_data)
            }
        }

    # 株式セクション
    stock_section = build_section(weekly_stock_data, monthly_stock_data)
    stock_section['lead_lag'] = stock_lead_lag

    # 債券セクション
    bond_section = build_section(weekly_bond_data, monthly_bond_data)
    bond_section['jgb_yield_correlation'] = bond_yield_correlation

    output = {
        'generated_at': datetime.now().isoformat(),
        'stock': stock_section,
        'bond': bond_section,
        # 後方互換性: 旧フォーマットのフィールドも維持
        'weekly': stock_section['weekly'],
        'monthly': stock_section['monthly'],
        'lead_lag': stock_lead_lag
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
    parser.add_argument('--skip-bond', action='store_true', help='Skip bond data fetching')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_path = output_dir / "foreign-investor.json"

    # 週次データ取得
    week_df = fetch_csv(WEEK_CSV_URL)
    if week_df is None:
        logging.error("Failed to fetch weekly data")
        return 1

    # 株式データ
    weekly_stock_data = parse_weekly_data(week_df)

    # 累積フローを事前計算（lead-lag分析に必要）
    recent_stock_weekly = weekly_stock_data[-52:] if len(weekly_stock_data) >= 52 else weekly_stock_data
    cumulative = 0
    for d in recent_stock_weekly:
        cumulative += d['net']
        d['cumulative'] = cumulative

    # 債券データ
    weekly_bond_data = []
    if not args.skip_bond:
        weekly_bond_data = parse_weekly_bond_data(week_df)
        # 債券の累積フローを計算
        recent_bond_weekly = weekly_bond_data[-52:] if len(weekly_bond_data) >= 52 else weekly_bond_data
        cumulative = 0
        for d in recent_bond_weekly:
            cumulative += d['net']
            d['cumulative'] = cumulative

    # 月次データ取得
    month_df = fetch_csv(MONTH_CSV_URL)
    monthly_stock_data = parse_monthly_data(month_df) if month_df is not None else []
    # 月次の債券データは別CSV（monthb3.csv）だが、今回は週次データのみを使用
    monthly_bond_data = []

    # 株式リード・ラグ分析（累積フローベース）
    stock_lead_lag = {'peak_lag': 0, 'peak_correlation': 0, 'interpretation': 'スキップ', 'correlations': []}
    if not args.skip_lead_lag:
        nikkei_df = load_nikkei_data()
        stock_lead_lag = calculate_lead_lag(recent_stock_weekly, nikkei_df)

    # 債券と金利の相関分析
    bond_yield_correlation = {'correlation': 0.0, 'yield_latest': None, 'interpretation': 'スキップ'}
    if not args.skip_bond and weekly_bond_data:
        jgb_yield_df = fetch_jgb_yield()
        recent_bond_weekly = weekly_bond_data[-52:] if len(weekly_bond_data) >= 52 else weekly_bond_data
        bond_yield_correlation = calculate_bond_yield_correlation(recent_bond_weekly, jgb_yield_df)

    # 出力生成
    output = generate_output(
        weekly_stock_data,
        monthly_stock_data,
        stock_lead_lag,
        weekly_bond_data,
        monthly_bond_data,
        bond_yield_correlation
    )

    # 保存
    save_output(output, output_path)

    # サマリー表示
    print(f"\n=== Foreign Investor Flow Data ===")
    print(f"Generated: {output['generated_at']}")

    # 株式サマリー
    print(f"\n[Stock]")
    print(f"Weekly data points: {len(output['stock']['weekly']['data'])}")
    if output['stock']['weekly']['latest']['date']:
        print(f"Latest: {output['stock']['weekly']['latest']['date']} - {output['stock']['weekly']['latest']['net_billion']:.0f}億円 (P{output['stock']['weekly']['latest']['percentile']:.0f})")
    print(f"52-week cumulative: {output['stock']['weekly']['cumulative_52w']:.0f}億円")
    print(f"Lead-lag: {output['stock']['lead_lag']['interpretation']}")

    # 債券サマリー
    if output['bond']['weekly']['data']:
        print(f"\n[Bond]")
        print(f"Weekly data points: {len(output['bond']['weekly']['data'])}")
        if output['bond']['weekly']['latest']['date']:
            print(f"Latest: {output['bond']['weekly']['latest']['date']} - {output['bond']['weekly']['latest']['net_billion']:.0f}億円 (P{output['bond']['weekly']['latest']['percentile']:.0f})")
        print(f"52-week cumulative: {output['bond']['weekly']['cumulative_52w']:.0f}億円")
        if output['bond']['jgb_yield_correlation']['yield_latest']:
            print(f"10Y JGB Yield: {output['bond']['jgb_yield_correlation']['yield_latest']}%")
        print(f"Yield correlation: {output['bond']['jgb_yield_correlation']['interpretation']}")

    return 0


if __name__ == '__main__':
    exit(main())
