import os
from datetime import date, datetime, timedelta

import pandas as pd

# Constants
from db_utils import TRANSACTIONS_CSV_PATH, DATE_FORMAT


def int_to_date(n: int, base_date: str = '2000-01-01') -> date:
    return datetime.strptime(base_date, DATE_FORMAT).date() + timedelta(n)


def month_delta(start_date: date, end_date: date) -> int:
    return 12 * (end_date.year - start_date.year) + (end_date.month - start_date.month)


def day_delta(start_date: date, end_date: date) -> int:
    return (end_date - start_date).days


def get_most_recent_business_day() -> str:
    # Get today's date
    today_date = date.today()
    # getting difference
    if today_date.weekday() == 6:  # Sunday
        diff = 2
    elif today_date.weekday() == 5:  # Saturday
        diff = 1
    else:
        diff = 0
    most_recent_business_day = today_date - timedelta(days=diff)
    return most_recent_business_day.strftime(DATE_FORMAT)


def ask_yes_no_question(question) -> str:
    yes_no_answer = input(question).upper()
    if yes_no_answer in ['YES', 'Y'] or yes_no_answer in ['NO', 'N']:
        return yes_no_answer
    else:
        return ask_yes_no_question(question)


def check_last_update_date(transactions_csv_path: str = TRANSACTIONS_CSV_PATH) -> str:
    # In case no transactions have been recorded yet, set 2000-01-01 as the last update date.
    if not os.path.exists(transactions_csv_path):
        return '2000-01-01'

    transactions_df = pd.read_csv(transactions_csv_path, index_col=[
        'id'], parse_dates=['date'])

    return transactions_df['date'].max().date().strftime(DATE_FORMAT)


def numeric_to_currency_str_format(n: float, currency_symbol: str = '€'):
    negative = n < 0
    if negative:
        return f'({currency_symbol}{-n:,.0f})'.replace(f'({currency_symbol}nan)', '')
    return f'{currency_symbol}{n:,.0f}'.replace(f'{currency_symbol}nan', '')


def numeric_to_percentage_str_format(n: float, precision: int = 1):
    return f'{n*100:,.{precision}f}%'.replace('nan%', '')


def numeric_to_bps_str_format(n: float):
    return f'{n*10000:,.0f}'.replace('nan', '')


def numeric_to_int_str_format(n: float):
    return f'{n:,.0f}'.replace('nan', '')


def numeric_to_float_str_format(n: float, precision: int = 1):
    return f'{n:,.{precision}f}'.replace('nan', '')


if __name__ == '__main__':
    pass
