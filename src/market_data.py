import os
import time
from datetime import datetime

import pandas as pd
import pandas_datareader.data as web
import requests
from numpy.random import rand

from db_utils import load_market_data_details
from utils import get_most_recent_business_day

from constants import MARKET_DATA_DETAILS_PATH, \
    MARKET_DATA_SERIES_PATH,\
    DATE_FORMAT, \
    MSCI_DATE_FORMAT


def retrieve_yahoo_adj_close_data(symbol: str,
                                  start_date: str,
                                  end_date: str = None,
                                  ) -> pd.Series:
    end_date = datetime.today().strftime(DATE_FORMAT) if end_date is None else end_date
    series = web.DataReader(symbol, 'yahoo', start=start_date, end=end_date)['Adj Close']
    return series.loc[~series.index.duplicated()]  # Avoid storing duplicates


def generate_returns_from_yahoo_data(yahoo_adj_close_data: pd.DataFrame):
    return yahoo_adj_close_data.interpolate().pct_change().dropna(how='all')


def build_msci_web_query(index_code: int,
                         currency_symbol: str,
                         index_variant: str,
                         start_date: str,
                         end_date: str = None,
                         data_frequency='DAILY') -> str:
    # Transform dates into MSCI Format
    end_date = get_most_recent_business_day() if end_date is None else end_date
    start_date = datetime.strptime(start_date, DATE_FORMAT).strftime(MSCI_DATE_FORMAT)
    end_date = datetime.strptime(end_date, DATE_FORMAT).strftime(MSCI_DATE_FORMAT)

    msci_query = f"https://app2.msci.com/products/service/index/indexmaster/getLevelDataForGraph?currency_symbol" \
                 f"={currency_symbol}&index_variant={index_variant}&start_date={start_date}&end_date={end_date}" \
                 f"&data_frequency={data_frequency}&index_codes={int(index_code)}"
    return msci_query


def retrieve_msci_data(index_code: int,
                       currency_symbol: str,
                       index_variant: str,
                       start_date: str,
                       end_date: str = None,
                       data_frequency='DAILY') -> pd.Series:
    # Build web query
    msci_query = build_msci_web_query(index_code=index_code,
                                      currency_symbol=currency_symbol,
                                      index_variant=index_variant,
                                      start_date=start_date,
                                      end_date=end_date,
                                      data_frequency=data_frequency)
    # Make a request to the MSCI website
    my_request = requests.get(msci_query)

    if my_request.status_code == 200:  # Request was successful
        df = pd.DataFrame.from_dict(my_request.json()['indexes']['INDEX_LEVELS'])
        df['calc_date'] = df['calc_date'].apply(lambda my_date: datetime.strptime(str(my_date), MSCI_DATE_FORMAT))
        df = df.rename({'calc_date': 'date'}, axis=1).set_index('date')
        series = df.iloc[:, 0]
        return series.loc[~series.index.duplicated()]  # Avoid storing duplicates
    else:
        print(f'The following web request:\n\n {msci_query}\n\n has failed with status code {my_request.status_code}')


# noinspection PyTypeChecker
def update_market_data_series_csv(market_data_details_path: str = MARKET_DATA_DETAILS_PATH,
                                  market_data_series_path: str = MARKET_DATA_SERIES_PATH):
    market_data_details = load_market_data_details(market_data_details_path)
    old_market_data_series = None
    series_id_to_last_update_date = {}

    if os.path.exists(market_data_series_path):
        # Read old data
        old_market_data_series = pd.read_csv(market_data_series_path, parse_dates=['date'], index_col='date')
        # Check latest update
        series_id_to_last_update_date = dict(old_market_data_series.apply(
            lambda col: col.dropna().index.tolist()[-1].strftime(DATE_FORMAT), axis=0))

    updated_series = []
    for series_id, row in market_data_details.iterrows():
        print(f"Updating {row['name']},"
              f" series_id: {series_id}")
        # Coerce series_id to be a string
        series_id = str(series_id)
        # Lat update date
        last_update = series_id_to_last_update_date.get(series_id, None)
        last_update =\
            last_update if last_update is not None else row['first_available_date'].date().strftime(DATE_FORMAT)

        # Yahoo Source
        if row['source'] == 'yahoo':
            series = retrieve_yahoo_adj_close_data(row['symbol'],
                                                   start_date=last_update,
                                                   end_date=None)
            time.sleep(2*rand())
            series.name = series_id
            updated_series.append(series)

        # MSCI Source
        elif row['source'] == 'msci':
            try:
                series = retrieve_msci_data(index_code=row['index_code'],
                                            currency_symbol=row['currency_symbol'],
                                            index_variant=row['index_variant'],
                                            data_frequency=row['data_frequency'],
                                            start_date=last_update,
                                            end_date=None)
                time.sleep(2*rand())
                series.name = series_id
                updated_series.append(series)
            except AttributeError:  # Request fails and returns None -> NoneType has no attribute name
                continue

    # Join the series
    new_market_data_series = pd.concat(updated_series, axis=1, join='outer')

    # Update
    if old_market_data_series is not None:
        new_market_data_series = new_market_data_series.combine_first(old_market_data_series).sort_index()

    new_market_data_series.rename_axis('date').to_csv(market_data_series_path)


if __name__ == '__main__':
    pass
