import getpass

import degiroapi
from utils import *

from db_utils import PRODUCT_DETAILS_CSV_PATH, DEGIRO_DATE_FORMAT


def start_degiro_session(username: str = None, password: str = None, login=True) -> degiroapi.DeGiro:
    # Initialize a DeGiro session
    degiro_session = degiroapi.DeGiro()

    # Get User Input and login
    if login:
        if username is None:
            print('Please type in your DeGiro Username and press enter: ')
            username = input()
        if password is None:
            print('Please type in your DeGiro Password and press enter: ')
            password = getpass.getpass()
        # Login
        degiro_session.login(username, password)

    return degiro_session


def generate_degiro_time_interval_update(start_date: str) -> (str,):
    # First generate date differences
    start_date = datetime.strptime(start_date, DATE_FORMAT).date()
    month_diff = month_delta(start_date, date.today())
    day_diff = day_delta(start_date, date.today())

    # Check how many data points we need in order to update the time series
    if day_diff < 1:  # No Update
        degiro_time_interval = None
    elif month_diff < 1 and day_diff < 7:  # Week
        degiro_time_interval = degiroapi.Interval.Type.One_Week
    elif month_diff < 1 and day_diff >= 7:  # One Month
        degiro_time_interval = degiroapi.Interval.Type.One_Month
    elif month_diff < 3:  # 3 Months
        degiro_time_interval = degiroapi.Interval.Type.Three_Months
    elif month_diff < 6:  # 6 Months
        degiro_time_interval = degiroapi.Interval.Type.Six_Months
    elif month_diff < 12:  # 1 year
        degiro_time_interval = degiroapi.Interval.Type.One_Year
    elif month_diff < 36:  # 3 years
        degiro_time_interval = degiroapi.Interval.Type.Three_Years
    elif month_diff < 60:  # 5 years
        degiro_time_interval = degiroapi.Interval.Type.Five_Years
    else:  # Max
        degiro_time_interval = degiroapi.Interval.Type.Max

    return degiro_time_interval


def retrieve_product_price_series(degiro_session: degiroapi.DeGiro,
                                  product_id_ls: list,
                                  start_date: str = '2000-01-01') -> pd.DataFrame:
    assert type(
        product_id_ls) == list, 'product_id_ls must be of type list.'

    # Recursive Case
    if len(product_id_ls) > 1:

        return pd.concat([retrieve_product_price_series(
            degiro_session, [product_id], start_date=start_date) for product_id in list(set(product_id_ls))], axis=0)

    # Base Case
    elif len(product_id_ls) == 1:
        product_id = int(product_id_ls[0])

        # Check how much data is needed to perform an update
        degiro_time_interval_update = generate_degiro_time_interval_update(start_date)

        # Retrieve the Historicall prices
        if degiro_time_interval_update is not None:
            real_time_price = degiro_session.real_time_price(
                product_id, degiro_time_interval_update)

            # Retrieve the base date
            base_date = datetime.strptime(
                real_time_price[0]['data']['windowStart'][:19], DEGIRO_DATE_FORMAT)
            base_date_str = base_date.strftime(DATE_FORMAT)
            # Construct Historicall Time-Series
            price_time_series = pd.DataFrame.from_dict(real_time_price[1]['data'])
            price_time_series.columns = ['date', 'price']
            # Int to Date
            price_time_series['date'] = price_time_series['date'].apply(
                lambda n: int_to_date(n, base_date_str))

            # Add product id field and sort
            price_time_series['product_id'] = product_id
            price_time_series = price_time_series[['product_id', 'date', 'price']]

        else:
            price_time_series = pd.DataFrame(columns=['product_id', 'date', 'price'])
        return price_time_series


def retrieve_degiro_transactions(degiro_session: degiroapi.DeGiro,
                                 start_date: str = '2000-01-01',
                                 end_date: str = None) -> pd.DataFrame:
    # Set Start and End Date
    start_date = datetime.strptime(start_date, DATE_FORMAT)
    end_date = datetime.today() if end_date is None else datetime.strptime(
        end_date, DATE_FORMAT)

    # Retrieve Transaction Data
    transactions = degiro_session.transactions(start_date, end_date)

    # No Transactions
    if len(transactions) < 1:
        transactions_df = pd.DataFrame()

    # 1 or more transactions
    else:
        transactions_df = pd.DataFrame.from_dict(transactions).set_index('id')
        # Only 1 transaction
        if transactions_df.shape[0] < 2:
            transactions_df['date'] = datetime.strptime(transactions_df['date'].values[0][:19],
                                                        DEGIRO_DATE_FORMAT).strftime(DATE_FORMAT)
        # More than 1 transaction
        else:
            transactions_df['date'] = transactions_df['date'].apply(
                lambda my_date: datetime.strptime(my_date[:19], DEGIRO_DATE_FORMAT).strftime(DATE_FORMAT))

    return transactions_df.rename({'productId': 'product_id',
                                   'orderTypeId': 'order_type_id',
                                   'counterParty': 'counterparty',
                                   'fxRate': 'fx_rate',
                                   'totalInBaseCurrency': 'total_in_base_currency',
                                   'totalPlusFeeInBaseCurrency': 'total_plus_fee_in_base_currency',
                                   'transactionTypeId': 'transaction_type_id',
                                   'tradingVenue': 'trading_venue',
                                   'feeInBaseCurrency': 'fee_in_base_currency'}, axis=1)


def retrieve_degiro_product_info(degiro_session: degiroapi.DeGiro, product_id_ls: list) -> pd.DataFrame:
    assert type(product_id_ls) == list, 'product_id_ls must be of type list.'

    # Base Case
    if len(product_id_ls) == 1:
        product_id = product_id_ls[0]
        product_info = degiro_session.product_info(int(product_id))
        fields_to_store = ['id', 'name', 'isin', 'symbol', 'currency',
                           'category', 'productType', 'productTypeId',
                           'exchangeId']

        return pd.DataFrame({field: [product_info.get(field, None)] for field in fields_to_store}). \
            rename({'id': 'product_id',
                    'name': 'degiro_name',
                    'productType': 'product_type',
                    'productTypeId': 'product_type_id',
                    'exchangeId': 'exchange_id'}, axis=1).set_index('product_id')

    # Recursive Case
    return pd.concat([retrieve_degiro_product_info(degiro_session, [idd]) for idd in list(set(product_id_ls))], axis=0)


def update_product_details_csv(degiro_session: degiroapi.DeGiro, product_id_ls: list,
                               products_csv_path: str = PRODUCT_DETAILS_CSV_PATH) -> None:
    product_id = set(product_id_ls)  # Remove Duplicates

    if os.path.exists(products_csv_path):
        old_products_df = pd.read_csv(products_csv_path, index_col='product_id')
        old_product_id = set(old_products_df.index.tolist())
        new_product_id = product_id - old_product_id

        if len(new_product_id) > 0:
            new_products_df = retrieve_degiro_product_info(degiro_session, list(new_product_id))
            products_df = pd.concat([old_products_df, new_products_df], axis=0)
            products_df.to_csv(products_csv_path)
    else:
        products_df = retrieve_degiro_product_info(degiro_session, list(product_id))
        products_df.to_csv(products_csv_path)


def update_transactions_csv(degiro_session: degiroapi.DeGiro,
                            transactions_csv_path: str = TRANSACTIONS_CSV_PATH,
                            start_date: str = '2000-01-01') -> None:
    # Retrieve latest transactions
    new_transactions_df = retrieve_degiro_transactions(degiro_session, start_date)

    # If file path exists -> load old file and update it
    if os.path.exists(transactions_csv_path):
        old_transactions_df = pd.read_csv(
            transactions_csv_path, index_col='id')

        new_transactions_df = old_transactions_df.combine_first(new_transactions_df)

    # Make sure product_id is saved with no decimals
    new_transactions_df.loc[:, 'product_id'] = new_transactions_df.loc[:, 'product_id'].astype(int)
    new_transactions_df.sort_values(by='date').to_csv(transactions_csv_path)


if __name__ == '__main__':
    pass
