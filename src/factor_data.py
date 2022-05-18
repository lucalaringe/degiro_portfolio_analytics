import pandas as pd
from numpy import nan
import pandas_datareader.data as web

from db_utils import load_factor_data_details

# Constants
from constants import FACTOR_DATA_DETAILS_PATH, FACTOR_DATA_SERIES_PATH


def update_ff_factor_data_series(factor_data_details_path: str = FACTOR_DATA_DETAILS_PATH) -> pd.DataFrame:
    # Load factor data details
    factor_data_details = load_factor_data_details(factor_data_details_path)
    ff_factor_data_details = factor_data_details.loc[factor_data_details['source'] == 'famafrench']
    data_tables_to_load = ff_factor_data_details.loc[:, 'search_keyword'].unique().tolist()
    
    factors = []
    for data_table in data_tables_to_load:
        data_table_factor_data_details = \
            ff_factor_data_details.loc[ff_factor_data_details['search_keyword'] == data_table]
        df = web.DataReader(data_table, "famafrench", start='1900-01-01')[0]
        df *= 0.01
        for series_id, row in data_table_factor_data_details.iterrows():
            print(f"Updating {row['name']}, series_id = {series_id}")
            series = df.loc[:, row['id_in_source']]
            factors.append(series.replace([-99.99, -999], nan))

    return pd.concat(factors, axis=1, join='outer')


def update_factor_data_series_csv(factor_data_details_path: str = FACTOR_DATA_DETAILS_PATH,
                                  factor_data_series_path: str = FACTOR_DATA_SERIES_PATH):

    # As of now only famafrench data
    ff_factor_data_series = update_ff_factor_data_series(factor_data_details_path=factor_data_details_path)
    ff_factor_data_series.rename_axis('date').to_csv(factor_data_series_path)


if __name__ == '__main__':
    pass
