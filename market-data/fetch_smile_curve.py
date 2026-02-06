"""
JPX日経225オプション価格情報からスマイルカーブデータを取得（8-mon-assets版）

データソース: https://svc.qri.jp/jpx/nkopm/
※リファラー制限があるため、JPXサイト経由でアクセスする必要がある
"""

import json
import os
import re
import logging
import time
import argparse
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Configuration
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"


def parse_number(text: str) -> float | None:
    """数値文字列をfloatに変換（カンマ、%記号を処理）"""
    if not text or text == '-' or text == '':
        return None
    # カンマと%を除去
    cleaned = text.replace(',', '').replace('%', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def fetch_smile_curve_data(headless: bool = True, max_retries: int = 3, num_months: int = 3) -> dict | None:
    """
    JPXオプション価格情報ページからスマイルカーブデータを取得

    Args:
        headless: ヘッドレスモード（デバッグ時はFalse）
        max_retries: 最大リトライ回数
        num_months: 取得する限月数（デフォルト3）

    Returns:
        dict: スマイルカーブデータ（失敗時はNone）
    """
    logging.info("Starting smile curve data fetch...")

    for attempt in range(max_retries):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=headless)
                page = browser.new_page()
                page.set_default_timeout(30000)

                # Step 1: JPXトップページにアクセス（リファラー確立）
                logging.info(f"Attempt {attempt + 1}/{max_retries}: Navigating to JPX derivatives quotes page...")
                page.goto("https://www.jpx.co.jp/markets/derivatives/quotes/index.html", wait_until="load", timeout=60000)

                # Step 2: オプション価格情報リンクをクリック
                logging.info("Clicking option price info link...")

                # 新しいタブで開くリンクをクリック
                with page.expect_popup() as popup_info:
                    page.click('a[href="https://svc.qri.jp/jpx/nkopm/"]')

                option_page = popup_info.value
                option_page.wait_for_load_state("load")

                logging.info(f"Option page loaded: {option_page.url}")

                # Step 3: 複数限月のデータを抽出
                result = extract_multiple_months_data(option_page, num_months)

                browser.close()

                if result:
                    logging.info("Smile curve data fetched successfully")
                    return result
                else:
                    logging.warning("Data extraction returned empty result")

        except PlaywrightTimeoutError as e:
            logging.warning(f"Attempt {attempt + 1} timed out: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    logging.error("Max retries reached. Fetch failed.")
    return None


def extract_multiple_months_data(page, num_months: int = 3) -> dict | None:
    """複数の限月タブからデータを抽出"""
    try:
        # 参考価格情報を取得（共通）
        underlying = extract_underlying_data(page)
        logging.info(f"Underlying data: {underlying}")

        # 限月タブを取得
        expiry_tabs = page.locator('ul >> li >> a:has-text("月限")').all()
        logging.info(f"Found {len(expiry_tabs)} expiry month tabs")

        expiry_months = []

        # 指定された数の限月を処理
        for i in range(min(num_months, len(expiry_tabs))):
            tab = expiry_tabs[i]
            tab_text = tab.inner_text()
            logging.info(f"Processing expiry tab {i + 1}: {tab_text}")

            # タブをクリック（最初のタブは既に選択されている可能性）
            if i > 0:
                tab.click()
                # データ更新を待機
                time.sleep(0.5)
                page.wait_for_load_state("networkidle", timeout=10000)

            # このタブのデータを抽出（タブのテキストを渡す）
            month_data = extract_single_month_data(page, underlying.get('futures'), tab_text)

            if month_data:
                expiry_months.append(month_data)
                logging.info(f"Extracted {len(month_data['data'])} data points for {month_data['label']}")

        if not expiry_months:
            return None

        result = {
            'timestamp': datetime.now().isoformat(),
            'underlying': underlying,
            'expiryMonths': expiry_months
        }

        return result

    except Exception as e:
        logging.error(f"Error extracting multiple months data: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_underlying_data(page) -> dict:
    """参考価格情報を抽出"""
    underlying = {}

    try:
        # 日経平均株価
        spot_cell = page.locator('table >> tr:has-text("日経平均株価") >> td:nth-child(2)').first
        if spot_cell.count() > 0:
            spot_text = spot_cell.inner_text()
            match = re.search(r'[\d,]+\.\d+', spot_text)
            if match:
                underlying['spot'] = parse_number(match.group())

        # 日経225先物
        futures_cell = page.locator('table >> tr:has-text("日経225先物") >> td:nth-child(2)').first
        if futures_cell.count() > 0:
            futures_text = futures_cell.inner_text()
            match = re.search(r'[\d,]+', futures_text)
            if match:
                underlying['futures'] = parse_number(match.group())

        # HV（ヒストリカルボラティリティ）
        hv_cell = page.locator('table >> tr:has-text("日経平均株価") >> td:nth-child(5)').first
        if hv_cell.count() > 0:
            hv_text = hv_cell.inner_text()
            underlying['hv'] = parse_number(hv_text)

    except Exception as e:
        logging.warning(f"Error extracting underlying data: {e}")

    return underlying


def extract_single_month_data(
    page, futures_price: float | None, tab_label: str | None = None
) -> dict | None:
    """単一限月のデータを抽出"""
    try:
        # タブラベルが渡された場合はそれを使用
        expiry_label = tab_label if tab_label else "不明"

        # "12月限月" → "202512"
        month_match = re.search(r'(\d+)月', expiry_label)
        current_year = datetime.now().year
        current_month = datetime.now().month

        if month_match:
            month = int(month_match.group(1))
            # 月が現在より小さければ来年
            if month < current_month:
                current_year += 1
            expiry_month = f"{current_year}{month:02d}"
        else:
            expiry_month = datetime.now().strftime('%Y%m')

        # オプションテーブルからデータを抽出
        smile_data = extract_table_data(page)

        if not smile_data:
            return None

        # ATMを特定
        atm_strike = None
        if futures_price:
            atm_strike = min(smile_data, key=lambda x: abs(x['strike'] - futures_price))['strike']

        return {
            'month': expiry_month,
            'label': expiry_label,
            'atmStrike': atm_strike,
            'data': sorted(smile_data, key=lambda x: x['strike'])
        }

    except Exception as e:
        logging.error(f"Error extracting single month data: {e}")
        return None


def extract_table_data(page) -> list:
    """テーブルからIVデータを抽出"""
    smile_data = []

    try:
        # テーブルを探す
        tables = page.locator('table').all()

        # メインのオプションテーブルを特定（行数が多いもの）
        main_table = None
        for table in tables:
            row_count = table.locator('tbody tr').count()
            if row_count > 10:
                main_table = table
                break

        if not main_table:
            for table in tables:
                row_count = table.locator('tr').count()
                if row_count > 10:
                    main_table = table
                    break

        if not main_table:
            return []

        rows = main_table.locator('tbody tr').all()
        if len(rows) == 0:
            rows = main_table.locator('tr').all()

        for row in rows:
            cells = row.locator('td').all()
            if len(cells) < 16:
                continue

            try:
                # 権利行使価格を探す
                strike_cell_idx = None
                for i, cell in enumerate(cells):
                    cell_text = cell.inner_text()
                    if 'リスク指標' in cell_text:
                        strike_match = re.search(r'([\d,]+)', cell_text)
                        if strike_match:
                            strike = parse_number(strike_match.group(1))
                            if strike and strike > 30000:
                                strike_cell_idx = i
                                break

                if strike_cell_idx is None:
                    continue

                strike = parse_number(re.search(r'([\d,]+)', cells[strike_cell_idx].inner_text()).group(1))

                # CALL IV
                call_iv = None
                for offset in [3, 4, 5]:
                    idx = strike_cell_idx - offset
                    if idx >= 0:
                        cell_text = cells[idx].inner_text()
                        lines = cell_text.strip().split('\n')
                        for line in lines:
                            if '%' in line:
                                iv_match = re.search(r'(\d+\.\d+)%', line)
                                if iv_match:
                                    iv_val = float(iv_match.group(1))
                                    if 10 <= iv_val <= 100:
                                        call_iv = iv_val
                                        break
                        if call_iv:
                            break

                # PUT IV
                put_iv = None
                for offset in [3, 4, 5]:
                    idx = strike_cell_idx + offset
                    if idx < len(cells):
                        cell_text = cells[idx].inner_text()
                        lines = cell_text.strip().split('\n')
                        for line in lines:
                            if '%' in line:
                                iv_match = re.search(r'(\d+\.\d+)%', line)
                                if iv_match:
                                    iv_val = float(iv_match.group(1))
                                    if 10 <= iv_val <= 100:
                                        put_iv = iv_val
                                        break
                        if put_iv:
                            break

                if strike and (call_iv or put_iv):
                    smile_data.append({
                        'strike': strike,
                        'callIV': call_iv,
                        'putIV': put_iv
                    })

            except Exception as e:
                continue

    except Exception as e:
        logging.error(f"Error extracting table data: {e}")

    return smile_data


def load_previous_data(prev_path: Path) -> dict | None:
    """前回のデータを読み込む"""
    try:
        if prev_path.exists():
            return json.loads(prev_path.read_text(encoding='utf-8'))
    except Exception as e:
        logging.warning(f"Failed to load previous data: {e}")
    return None


def calculate_iv_changes(current_data: dict, prev_data: dict | None) -> dict:
    """IV変化を計算してデータに追加"""
    if not prev_data:
        return current_data

    # 前回データをストライク価格でインデックス化
    prev_by_month = {}
    for month in prev_data.get('expiryMonths', []):
        month_key = month['month']
        prev_by_month[month_key] = {
            d['strike']: d for d in month.get('data', [])
        }

    # 各限月のIV変化を計算
    for month in current_data.get('expiryMonths', []):
        month_key = month['month']
        prev_strikes = prev_by_month.get(month_key, {})

        for data_point in month.get('data', []):
            strike = data_point['strike']
            prev_point = prev_strikes.get(strike, {})

            # CALL IV変化
            if data_point.get('callIV') is not None and prev_point.get('callIV') is not None:
                data_point['callIVChange'] = round(data_point['callIV'] - prev_point['callIV'], 2)
            else:
                data_point['callIVChange'] = None

            # PUT IV変化
            if data_point.get('putIV') is not None and prev_point.get('putIV') is not None:
                data_point['putIVChange'] = round(data_point['putIV'] - prev_point['putIV'], 2)
            else:
                data_point['putIVChange'] = None

    return current_data


def save_data(data: dict, output_path: Path, prev_path: Path):
    """データをJSONファイルに保存（前回データも保持）"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 現在のデータを前回データとしてバックアップ（存在する場合）
    if output_path.exists():
        try:
            import shutil
            shutil.copy2(output_path, prev_path)
            logging.info(f"Previous data backed up to: {prev_path}")
        except Exception as e:
            logging.warning(f"Failed to backup previous data: {e}")

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    logging.info(f"Data saved to: {output_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='Fetch smile curve data from JPX')
    parser.add_argument('--no-headless', action='store_true', help='Run browser in visible mode')
    parser.add_argument('--output-dir', type=str, help='Output directory path')
    parser.add_argument('--months', type=int, default=3, help='Number of expiry months to fetch (default: 3)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_path = output_dir / "smile-curve.json"
    prev_path = output_dir / "smile-curve-prev.json"

    data = fetch_smile_curve_data(headless=not args.no_headless, num_months=args.months)

    if data:
        # 前回データを読み込んでIV変化を計算
        prev_data = load_previous_data(prev_path)
        data = calculate_iv_changes(data, prev_data)

        save_data(data, output_path, prev_path)
        print(f"\nSuccess! Data saved to: {output_path}")
        print(f"  - Timestamp: {data['timestamp']}")
        print(f"  - Underlying: {data['underlying']}")
        print(f"  - Expiry months: {len(data['expiryMonths'])}")
        for month in data['expiryMonths']:
            # IV変化があるデータポイント数をカウント
            changes = sum(1 for d in month['data'] if d.get('callIVChange') or d.get('putIVChange'))
            print(f"    - {month['label']}: {len(month['data'])} data points (ATM: {month['atmStrike']}, IV changes: {changes})")
    else:
        print("\nFailed to fetch data")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
