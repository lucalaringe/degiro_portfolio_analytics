import os

# Constants
DEGIRO_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'
MSCI_DATE_FORMAT = '%Y%m%d'

PORTFOLIO_DATA_FOLDER_PATH = 'data/portfolio_data'
if not os.path.exists(PORTFOLIO_DATA_FOLDER_PATH):
    os.mkdir(PORTFOLIO_DATA_FOLDER_PATH)
MARKET_DATA_FOLDER_PATH = 'data/market_data'
if not os.path.exists(MARKET_DATA_FOLDER_PATH):
    os.mkdir(MARKET_DATA_FOLDER_PATH)
FACTOR_DATA_FOLDER_PATH = 'data/factor_data'
if not os.path.exists(FACTOR_DATA_FOLDER_PATH):
    os.mkdir(FACTOR_DATA_FOLDER_PATH)

TRANSACTIONS_CSV_PATH = os.path.join(PORTFOLIO_DATA_FOLDER_PATH, 'transactions.csv')
PRODUCT_DETAILS_CSV_PATH = os.path.join(PORTFOLIO_DATA_FOLDER_PATH, 'product_details.csv')
PRODUCT_PRICE_SERIES_CSV_PATH = os.path.join(PORTFOLIO_DATA_FOLDER_PATH, 'product_price_series.csv')
QUOTAS_CSV_PATH = os.path.join(PORTFOLIO_DATA_FOLDER_PATH, 'quotas.csv')
NAV_CSV_PATH = os.path.join(PORTFOLIO_DATA_FOLDER_PATH, 'nav.csv')

MARKET_DATA_DETAILS_PATH = os.path.join(MARKET_DATA_FOLDER_PATH, 'market_data_details.csv')
MARKET_DATA_SERIES_PATH = os.path.join(MARKET_DATA_FOLDER_PATH, 'market_data_series.csv')
FACTOR_DATA_DETAILS_PATH = os.path.join(FACTOR_DATA_FOLDER_PATH, 'factor_data_details.csv')
FACTOR_DATA_SERIES_PATH = os.path.join(FACTOR_DATA_FOLDER_PATH, 'factor_data_series.csv')

AVAILABLE_CURRENCIES = {'Local', 'EUR', 'USD', 'BTC', 'JPY'}


if __name__ == '__main__':
    pass
