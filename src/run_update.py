from degiro_utils import *
from market_data import update_market_data_series_csv
from factor_data import update_factor_data_series_csv
from just_etf_scraping import augment_product_details_csv
from db_utils import *


def run_update_transactions_csv(degiro_session: degiroapi.DeGiro,
                                start_date: str = '2000-01-01',
                                transactions_csv_path: str = TRANSACTIONS_CSV_PATH) -> None:
    exit_loop = 'NO'

    while exit_loop == 'NO':
        # Update the transactions
        print('Updating the transactions csv...\n')
        update_transactions_csv(
            degiro_session=degiro_session,
            transactions_csv_path=transactions_csv_path,
            start_date=start_date)
        print('Done.\n')
        # Ask if another update from another account is needed
        exit_loop = ask_yes_no_question(
            'Type YES if you want the routine to move forward to the Historical data retrieval.\n\nIn case you wish to '
            'repeat the transactions csv update procedure for a different account, type NO. (y/n)... ')

        if exit_loop == 'NO':
            # Logout and start a new DeGiro session
            degiro_session.logout()
            degiro_session = start_degiro_session()


def run_update_product_price_series_csv(degiro_session: degiroapi.DeGiro,
                                        transactions_df: pd.DataFrame,
                                        product_price_series_csv_path: str = PRODUCT_PRICE_SERIES_CSV_PATH) -> None:
    product_id_ls = list(set(transactions_df['product_id'].tolist()))
    old_price_time_series = None  # Initialize

    if os.path.exists(product_price_series_csv_path):
        old_price_time_series = pd.read_csv(
            product_price_series_csv_path, parse_dates=['date']).sort_values(['product_id', 'date'], ascending=True)
        product_id_to_last_update_date = old_price_time_series.groupby(['product_id'])[['date']].last()['date'].\
            apply(lambda dt: dt.strftime(DATE_FORMAT)).to_dict()
        # Date to String
        old_price_time_series['date'] = old_price_time_series['date'].apply(lambda dt: dt.strftime(DATE_FORMAT))
    else:
        # Load full history
        product_id_to_last_update_date = dict(zip(product_id_ls, ['1900-01-01'] * len(product_id_ls)))

    # Loop through products and store new series
    updated_series = []
    for product_id in product_id_to_last_update_date.keys():
        updated_series.append(
            retrieve_product_price_series(degiro_session,
                                          [product_id],
                                          start_date=product_id_to_last_update_date.get(product_id, '1900-01-01')))
    new_price_time_series = pd.concat(updated_series, axis=0)
    # Date to String
    new_price_time_series['date'] = new_price_time_series['date'].apply(lambda dt: dt.strftime(DATE_FORMAT))

    if os.path.exists(product_price_series_csv_path):  # Merge with old data
        new_price_time_series = pd.concat([new_price_time_series, old_price_time_series], axis=0).drop_duplicates()

    # Write csv
    new_price_time_series.sort_values(['product_id', 'date', 'price']).to_csv(product_price_series_csv_path,
                                                                              index=False)


def run_update(username: str = None,
               password: str = None,
               start_date: str = None,
               transactions_csv_path: str = TRANSACTIONS_CSV_PATH,
               product_details_csv_path: str = PRODUCT_DETAILS_CSV_PATH,
               product_price_series_csv_path: str = PRODUCT_PRICE_SERIES_CSV_PATH,
               ) -> None:
    # Retrieve last update date as a default
    start_date = check_last_update_date(transactions_csv_path) if start_date is None else start_date

    # Start a DeGiro session
    print('Welcome!\n\n')
    degiro_session = start_degiro_session(username=username, password=password)

    enter_loop = ask_yes_no_question('Do you want to update your DeGiro transactions? (y/n)...')

    # Update Transaction Data
    if enter_loop in ['Y', 'YES']:
        run_update_transactions_csv(degiro_session=degiro_session,
                                    start_date=start_date,
                                    transactions_csv_path=transactions_csv_path)

    # Load Transaction data
    transactions_df = load_transactions()

    enter_loop = ask_yes_no_question('Do you want to update your DeGiro Product Details? (y/n)...')
    if enter_loop in ['Y', 'YES']:
        print('Updating the product details csv...\n')
        update_product_details_csv(
            degiro_session, transactions_df['product_id'].tolist(), product_details_csv_path)
        augment_product_details_csv(product_details_csv_path=product_details_csv_path)
        print('Done.\n')

    print('Retrieving and storing Historical daily close prices...\n')
    # Prices
    run_update_product_price_series_csv(degiro_session,
                                        transactions_df,
                                        product_price_series_csv_path)

    # Log out from DeGiro
    degiro_session.logout()

    enter_loop = ask_yes_no_question('Do you want to update market data? (y/n)...')
    if enter_loop in ['Y', 'YES']:
        print('Updating market data series csv...\n')
        update_market_data_series_csv(market_data_details_path=MARKET_DATA_DETAILS_PATH,
                                      market_data_series_path=MARKET_DATA_SERIES_PATH)
        print('Done.\n')

    enter_loop = ask_yes_no_question('Do you want to update factor data? (y/n)...')
    if enter_loop in ['Y', 'YES']:
        print('Updating factor data series csv...\n')
        update_factor_data_series_csv(factor_data_details_path=FACTOR_DATA_DETAILS_PATH,
                                      factor_data_series_path=FACTOR_DATA_SERIES_PATH)
        print('Done.\n')

    print('Update is finished. Exiting Routine...')


if __name__ == '__main__':
    run_update()
