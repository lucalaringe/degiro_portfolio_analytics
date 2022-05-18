import numpy as np
import pandas as pd
from numpy import nan

from constants import *
import nav_utils as nav


def load_product_details(product_details_csv_path: str = PRODUCT_DETAILS_CSV_PATH) -> pd.DataFrame:
    """
    Loads the product details csv in a pandas dataframe. The dataframe contains the following columns:
    ['degiro_name', 'isin', 'symbol', 'currency', 'category', 'product_type',
       'product_type_id', 'exchange_id', 'just_etf_name', 'ticker',
       'description', 'benchmark', 'asset_class', 'replication',
       'just_etf_currency', 'fund_size', 'fund_size_category', 'currency_risk',
       'inception_date', 'ter', 'distribution_policy', 'fund_domicile',
       'fund_structure', 'fund_provider', 'administrator', 'last_update',
       'name']

    :param product_details_csv_path: relative file path of the product details csv
    :return: product details dataframe
    """
    return pd.read_csv(product_details_csv_path, index_col='product_id').dropna(how='all')


def generate_product_id_to_field_dict(product_details_csv_path: str = PRODUCT_DETAILS_CSV_PATH,
                                      mapping_field: str = 'name') -> dict:
    """
    Returns a dictionary that maps product_id to mapping_field.

    :param product_details_csv_path: relative file path of the product details csv
    :param mapping_field: field of the product details table to which the mapping should lead to
    :return: dictionary that maps product_id to mapping_field
    """
    product_details = load_product_details(product_details_csv_path=product_details_csv_path).reset_index(drop=False)
    groupby_field_to_groupby_field = dict(zip(product_details.loc[:, 'product_id'],
                                                   product_details.loc[:, mapping_field]))
    return groupby_field_to_groupby_field


def load_market_data_details(market_data_details_path: str = MARKET_DATA_DETAILS_PATH) -> pd.DataFrame:
    """
    Loads the market data details csv file in memory in the form of a Pandas.DataFrame.
    The data fields are the following:
    ['name', 'source', 'symbol', 'index_code', 'currency_symbol',
       'index_variant', 'data_frequency', 'type', 'first_available_date',
       'is_exchange_rate']

    :param market_data_details_path: relative file path of the market data details csv
    :return: market data details dataframe
    """
    return pd.read_csv(market_data_details_path,
                       index_col='series_id',
                       parse_dates=['first_available_date']).dropna(how='all')


def convert_price_series(series: pd.Series,
                         start_currency: str,
                         end_currency: str,
                         exchange_rate_df: pd.DataFrame) -> pd.Series:
    """
    Converts a price series from one currency to another. Currently the currencies supported are USD, EUR, JPY and BTC

    :param series: price series
    :param start_currency: start currency code (e.g. 'USD')
    :param end_currency: end currency code (e.g. 'USD')
    :param exchange_rate_df: dataframe containing exchange rate time series
    :return: converted price series
    """
    if start_currency == end_currency:
        return series

    # First Convert Currency to USD
    # {'1005': ('USD', 'EUR'),
    # '1006': ('BTC', 'USD'),
    # '1013': ('JPY', 'USD')}

    if start_currency == 'EUR':
        series = series.multiply(exchange_rate_df.loc[:, '1005']).dropna()
    elif start_currency == 'JPY':
        series = series.multiply(1 / exchange_rate_df.loc[:, '1013']).dropna()
    elif start_currency == 'BTC':
        series = series.multiply(1 / exchange_rate_df.loc[:, '1006']).dropna()

    # Convert to end currency
    if end_currency == 'EUR':
        series = series.multiply(1 / exchange_rate_df.loc[:, '1005']).dropna()
    elif end_currency == 'JPY':
        series = series.multiply(exchange_rate_df.loc[:, '1013']).dropna()
    elif end_currency == 'BTC':
        series = series.multiply(exchange_rate_df.loc[:, '1006']).dropna()

    return series


def load_market_data_series(market_data_details_path: str = MARKET_DATA_DETAILS_PATH,
                            market_data_series_path: str = MARKET_DATA_SERIES_PATH,
                            currency='Local',
                            mapping_field: str = 'name',
                            include_exchange_rates: bool = True,
                            frequency: str = '1D'
                            ) -> pd.DataFrame:
    """
    Loads the market price series tracked in memory. Market data series are used for portfolio benchmarking purposes.
    Examples of market price series include 'Nikkei 225', 'Russel 2000' and 'S&P 500'.

    :param market_data_details_path: relative file path of the market data details csv
    :param market_data_series_path: relative file path of the market data series csv
    :param currency: specify which currency the market dta series should be in. If 'Local', each series is reported
    if its original currency (e.g. S&P500 will be reported in USD)
    :param mapping_field: which id field to report. Must be in {'series_id', 'name'}
    :param include_exchange_rates: indicates whether to include exchange rates
    :param frequency: time-series frequency
    :return: dataframe of market data price series
    """

    assert currency in AVAILABLE_CURRENCIES, f"Error! currency must be in {AVAILABLE_CURRENCIES}"
    assert mapping_field in {'series_id', 'name'}, "Error! groupby_field must be in in {'series_id', 'name'}"

    data = pd.read_csv(market_data_series_path, index_col='date', parse_dates=['date'])
    data_details = load_market_data_details(market_data_details_path=market_data_details_path)
    data = data.resample(frequency).last().dropna(how='all')

    if currency != 'Local':
        market_data_df = data.loc[:, data_details.loc[data_details['is_exchange_rate'] == 0].index.astype(str)]
        exchange_rate_df = data.loc[:, data_details.loc[data_details['is_exchange_rate'] == 1].index.astype(str)]
        market_data_df = market_data_df.resample(frequency).last().dropna(how='all')
        exchange_rate_df = exchange_rate_df.resample(frequency).last().dropna(how='all')

        market_data_df = market_data_df.apply(
            lambda col: convert_price_series(series=col,
                                             start_currency=data_details.loc[int(col.name), 'currency_symbol'],
                                             end_currency=currency,
                                             exchange_rate_df=exchange_rate_df), axis=0
        )
        # Join ER and sort columns
        data = market_data_df.join(exchange_rate_df)
        data = data[sorted(data.columns)]

    if not include_exchange_rates:
        data.drop(['1005', '1006', '1013'], axis=1, inplace=True)

    if mapping_field == 'name':
        id_to_name = dict(zip(data_details.index.astype(str), data_details.loc[:, 'name']))
        data.rename(id_to_name, axis=1, inplace=True)
        data = data[sorted(data.columns)]  # Sort in alphabetic order

    return data


def load_factor_data_details(factor_data_details_path: str = FACTOR_DATA_DETAILS_PATH) -> pd.DataFrame:
    """
    Loads in memory the content of the factor data details csv file. The data fields are:
    ['name', 'id_in_source', 'source', 'search_keyword', 'data_frequency',
       'first_available_date', 'multiplier', 'type']

    :param factor_data_details_path: relative path of the factor data details file.
    :return: dataframe of factor details
    """
    return pd.read_csv(factor_data_details_path,
                       index_col='series_id',
                       parse_dates=['first_available_date']).dropna(how='all')


def load_factor_data_series(factor_data_details_path: str = FACTOR_DATA_DETAILS_PATH,
                            factor_data_series_path: str = FACTOR_DATA_SERIES_PATH) -> pd.DataFrame:
    """
    Loads the factor data time-series in memory. Currently, the following factors are supported:
    ['Mkt-RF', 'Size', 'Value', 'RF', 'Profitability', 'Investment',
       'Momentum', 'Short-term Reversal', 'Long-term Reversal']

    :param factor_data_details_path: relative path of the factor data details file.
    :param factor_data_series_path: relative path of the factor data series csv
    :return: wide form dataframe of factor time series
    """
    details = load_factor_data_details(factor_data_details_path=factor_data_details_path).reset_index(drop=False)
    column_mapping = dict(zip(details['id_in_source'], details['name']))
    data = pd.read_csv(factor_data_series_path, index_col='date', parse_dates=['date'])
    return data.rename(column_mapping, axis=1)


def load_product_price_time_series(product_details_csv_path: str = PRODUCT_DETAILS_CSV_PATH,
                                   product_price_series_csv_path: str = PRODUCT_PRICE_SERIES_CSV_PATH,
                                   mapping_field: str = 'name',
                                   frequency='1D',
                                   ffill_non_overlapping_bank_holidays: bool = True,
                                   unprocessed_long_form: bool = False) -> pd.DataFrame:
    """
    Loads the product price time series in memory. Such series were previously extracted from Degiro via the product id
    and saved onto the product price time-series file.

    :param product_details_csv_path: relative path of the product details csv file
    :param product_price_series_csv_path: relative path of the product price series csv file
    :param mapping_field: field to identify the series (e.g. 'product_id', 'name', etc.).
    :param frequency: time-series frequency
    :param ffill_non_overlapping_bank_holidays: defaults to True. Effectively removes 'holes' in the time series. For
     instance, there might be some dates which are not available for most product (e.g. Christmas). If this
     argument is set to True the latter will be dropped from the time-series.
    :param unprocessed_long_form: indicates whether to return he data in long form, with no change in frequency or
     mapping field
    :return: wide form product time-series dataframe.
    """

    product_price_series = pd.read_csv(product_price_series_csv_path, parse_dates=['date'])
    product_price_series['product_id'] = product_price_series['product_id'].astype(int)
    product_price_series = pd.pivot_table(product_price_series, index='date', values='price', columns='product_id')

    if ffill_non_overlapping_bank_holidays:
        # First, adjust time-series starting date
        def _find_starting_date(series: pd.Series, min_consecutive_days: int = 30):
            int_nan_series = series.isna().astype(int)
            dates = series.index
            starting_date = int_nan_series.loc[int_nan_series.rolling(min_consecutive_days).mean() == 0].index[0]
            idx = np.where(dates == starting_date)[0][0]
            return (dates[idx-30+1]).date().strftime(DATE_FORMAT)

        def _adjust_series_starting_date(series: pd.Series):
            series[:_find_starting_date(series)] = nan
            return series

        product_price_series = product_price_series.apply(_adjust_series_starting_date, axis=0)
        product_price_series = product_price_series.ffill()

    if unprocessed_long_form:
        return product_price_series.melt(ignore_index=False, value_name='price').reset_index(drop=False)

    product_price_series = product_price_series.resample(frequency).last().dropna(how='all')
    product_id_to_field_dict = generate_product_id_to_field_dict(product_details_csv_path=product_details_csv_path,
                                                                 mapping_field=mapping_field)
    return product_price_series.rename(product_id_to_field_dict, axis=1)


def load_quotas_time_series(product_details_csv_path: str = PRODUCT_DETAILS_CSV_PATH,
                            transactions_csv_path: str = TRANSACTIONS_CSV_PATH,
                            mapping_field: str = 'name',
                            frequency='1D',
                            unprocessed_long_form: bool = False) -> pd.DataFrame:
    """
    Generates and load product 'quotas' (essentially units) in memory.

    :param product_details_csv_path: relative path of the product details csv file
    :param transactions_csv_path: relative path of the transactions csv file
    :param mapping_field: field to identify the series (e.g. 'product_id', 'name', etc.).
    :param frequency: time-series frequency
    :param unprocessed_long_form: indicates whether to return he data in long form, with no change in frequency or
     mapping field
    :return: quotas dataframe
    """
    transactions_df = load_transactions(transactions_csv_path=transactions_csv_path)
    quotas_df = nav.generate_quotas(transactions_df)
    if unprocessed_long_form:
        return quotas_df
    quotas_df['product_id'] = quotas_df['product_id']
    quotas_df = pd.pivot_table(quotas_df, index='date', values='quota', columns='product_id')
    quotas_series = quotas_df.resample(frequency).last().dropna(how='all')
    product_id_to_field_dict = generate_product_id_to_field_dict(product_details_csv_path=product_details_csv_path,
                                                                 mapping_field=mapping_field)
    return quotas_series.rename(product_id_to_field_dict, axis=1)


def load_product_return_time_series(product_details_csv_path: str = PRODUCT_DETAILS_CSV_PATH,
                                    mapping_field: str = 'name',
                                    frequency='1D',
                                    ffill_non_overlapping_bank_holidays: bool = True,
                                    ) -> pd.DataFrame:
    """
    Loads the product return time-series in memory.

    :param product_details_csv_path: relative path of the product details csv file
    :param mapping_field: field to identify the series (e.g. 'product_id', 'name', etc.).
    :param frequency: time-series frequency
    :param ffill_non_overlapping_bank_holidays: defaults to True. Effectively removes 'holes' in the time series. For
     instance, there might be some dates which are not available for most product (e.g. Christmas). If this
     argument is set to True the latter will be dropped from the time-series.
    :return: wide form return dataframe
    """
    product_price_series = load_product_price_time_series(
        mapping_field='product_id',
        ffill_non_overlapping_bank_holidays=ffill_non_overlapping_bank_holidays)
    product_daily_return_time_series = product_price_series.pct_change().dropna(how='all')
    product_return_time_series = product_daily_return_time_series.resample(frequency).\
        apply(lambda col: (1 + col).prod() - 1).replace(0, nan).dropna(how='all')

    product_id_to_field_dict = generate_product_id_to_field_dict(product_details_csv_path=product_details_csv_path,
                                                                 mapping_field=mapping_field)
    return product_return_time_series.rename(product_id_to_field_dict, axis=1)


def load_factor_return_series(factor_data_series_path=FACTOR_DATA_SERIES_PATH,
                              frequency='1W') -> pd.DataFrame:  # Defaults to weekly
    """
    Loads wide-form factor returns dataframe in memory.

    :param factor_data_series_path: relative path of the factor data series csv file
    :param frequency: time-series frequency (1W is the highest frequency)
    :return: factor returns dataframe
    """
    data = load_factor_data_series(factor_data_series_path=factor_data_series_path)
    return data.resample(frequency).apply(lambda series: (1 + series).prod() - 1).replace(0, nan)


def load_market_return_series(market_data_details_path=MARKET_DATA_DETAILS_PATH,
                              market_data_series_path=MARKET_DATA_SERIES_PATH,
                              frequency='1M',  # Defaults to Monthly
                              currency='Local') -> pd.DataFrame:
    """
    Load wide-form market data series returns dataframe.

    :param market_data_details_path: relative path of the market data details csv file
    :param market_data_series_path: relative path of the market data series csv file
    :param frequency: time-series frequency
    :param currency: specify which currency the market dta series should be in. If 'Local', each series is reported
    if its original currency (e.g. S&P500 will be reported in USD)
    :return: wide-form market data returns dataframe
    """
    assert currency in AVAILABLE_CURRENCIES, f"Error! currency must be in {AVAILABLE_CURRENCIES}"

    data = load_market_data_series(market_data_details_path=market_data_details_path,
                                   market_data_series_path=market_data_series_path,
                                   currency=currency,
                                   frequency=frequency,
                                   mapping_field='name',
                                   include_exchange_rates=False)
    return data.pct_change().dropna(how='all').replace(0, nan).dropna(axis=1, how='all')


def load_transactions(transactions_csv_path: str = TRANSACTIONS_CSV_PATH) -> pd.DataFrame:
    """
    Loads the portfolio transactions in memory. Transactions are originally extracted from the user's Degiro account.
    The data fields are:
    ['autoFxFeeInBaseCurrency', 'buysell', 'comments', 'counterparty',
           'date', 'fee_in_base_currency', 'fx_rate', 'grossFxRate', 'nettFxRate',
           'order_type_id', 'price', 'product_id', 'quantity', 'total',
           'totalFeesInBaseCurrency', 'totalPlusAllFeesInBaseCurrency',
           'total_in_base_currency', 'total_plus_fee_in_base_currency',
           'trading_venue', 'transaction_type_id', 'transfered']

    :param transactions_csv_path: relative path of the transactions csv file
    :return: transactions dataframe
    """
    return pd.read_csv(transactions_csv_path,
                       index_col='id',
                       parse_dates=['date']).sort_values(by='date')


def load_portfolio_asset_cash_flows(product_details_csv_path=PRODUCT_DETAILS_CSV_PATH,
                                    transactions_csv_path: str = TRANSACTIONS_CSV_PATH,
                                    mapping_field='name',
                                    frequency: str = '1D',
                                    net: bool = True,
                                    unprocessed_long_form: bool = False) -> pd.DataFrame:
    """
    Loads a wide-form dataframe of cash flows to the investments.

    :param product_details_csv_path: relative path of the product details csv file
    :param transactions_csv_path: relative path of the transactions csv file
    :param mapping_field: field to identify the series (e.g. 'product_id', 'name', etc.).
    :param frequency: time-series frequency
    :param net: indicates whether the cash flows should be net of fees
    :param unprocessed_long_form: indicates whether to return he data in long form, with no change in frequency or
     mapping field
    :return: wide-form cash flows dataframe
    """
    transactions_df = load_transactions(transactions_csv_path=transactions_csv_path)
    cash_flows_df = nav.generate_cash_flows(transactions_df=transactions_df,
                                            net=net)
    if unprocessed_long_form:
        return cash_flows_df

    cash_flows_df = pd.pivot_table(cash_flows_df, index='date', values='cf', columns='product_id')
    cash_flows_df = cash_flows_df.resample(frequency).sum().dropna(how='all')
    product_id_to_field_dict = generate_product_id_to_field_dict(product_details_csv_path=product_details_csv_path,
                                                                 mapping_field=mapping_field)
    return cash_flows_df.rename(product_id_to_field_dict, axis=1)


def load_portfolio_cash_flows(product_details_csv_path=PRODUCT_DETAILS_CSV_PATH,
                              transactions_csv_path: str = TRANSACTIONS_CSV_PATH,
                              frequency: str = '1D',
                              net: bool = True,
                              unprocessed_long_form: bool = False) -> pd.Series:
    """
    Loads the series of portfolio cash flows to the investment.

    :param product_details_csv_path: relative path of the product details csv file
    :param transactions_csv_path: relative path of the transactions csv file
    :param frequency: time-series frequency
    :param net: indicates whether the cash flows should be net of fees
    :param unprocessed_long_form: indicates whether to return he data in long form, with no change in frequency or
     mapping field
    :return: series of portfolio cash flows
    """
    if unprocessed_long_form:
        portfolio_asset_cash_flows = load_portfolio_asset_cash_flows(product_details_csv_path=product_details_csv_path,
                                                                     transactions_csv_path=transactions_csv_path,
                                                                     mapping_field='product_id',
                                                                     frequency=frequency,
                                                                     net=net,
                                                                     unprocessed_long_form=unprocessed_long_form)
        return nav.generate_portfolio_cash_flow(portfolio_asset_cash_flows)

    portfolio_asset_cash_flows = load_portfolio_asset_cash_flows(product_details_csv_path=product_details_csv_path,
                                                                 transactions_csv_path=transactions_csv_path,
                                                                 mapping_field='product_id',
                                                                 frequency=frequency,
                                                                 net=net
                                                                 )
    return portfolio_asset_cash_flows.sum(axis=1).rename('portfolio_cf')


def load_portfolio_asset_fees(product_details_csv_path=PRODUCT_DETAILS_CSV_PATH,
                              transactions_csv_path: str = TRANSACTIONS_CSV_PATH,
                              mapping_field='name',
                              frequency: str = '1D',
                              unprocessed_long_form: bool = False) -> pd.DataFrame:
    """
    Loads a wide-form dataframe of fees paid per investments.

    :param product_details_csv_path: relative path of the product details csv file
    :param transactions_csv_path: relative path of the transactions csv file
    :param mapping_field: field to identify the series (e.g. 'product_id', 'name', etc.).
    :param frequency: time-series frequency
    :param unprocessed_long_form: indicates whether to return he data in long form, with no change in frequency or
     mapping field
    :return: dataframe of fees paid per investments
    """
    transactions_df = load_transactions(transactions_csv_path=transactions_csv_path)
    fees_df = nav.generate_fees(transactions_df=transactions_df)
    if unprocessed_long_form:
        return fees_df

    fees_df['product_id'] = fees_df['product_id'].astype(int)
    fees_df = pd.pivot_table(fees_df, index='date', values='fee', columns='product_id')
    fees_df = fees_df.resample(frequency).sum().dropna(how='all')
    product_id_to_field_dict = generate_product_id_to_field_dict(product_details_csv_path=product_details_csv_path,
                                                                 mapping_field=mapping_field)
    return fees_df.rename(product_id_to_field_dict, axis=1)


def load_portfolio_fees(product_details_csv_path=PRODUCT_DETAILS_CSV_PATH,
                        transactions_csv_path: str = TRANSACTIONS_CSV_PATH,
                        frequency: str = '1D'
                        ) -> pd.Series:
    """
    Loads a series of fees paid for the whole portfolio.

    :param product_details_csv_path: relative path of the product details csv file
    :param transactions_csv_path: relative path of the transactions csv file
    :param frequency: time-series frequency

    :return: series of fees paid for the whole portfolio
    """
    portfolio_asset_fees = load_portfolio_asset_fees(product_details_csv_path=product_details_csv_path,
                                                     transactions_csv_path=transactions_csv_path,
                                                     mapping_field='product_id',
                                                     frequency=frequency
                                                     )
    return portfolio_asset_fees.sum(axis=1).rename('portfolio_fee')


def load_portfolio_asset_nav(product_details_csv_path: str = PRODUCT_DETAILS_CSV_PATH,
                             transactions_csv_path: str = TRANSACTIONS_CSV_PATH,
                             product_price_series_csv_path: str = PRODUCT_PRICE_SERIES_CSV_PATH,
                             groupby_field='name',
                             frequency: str = '1D',
                             unprocessed_long_form: bool = False) -> pd.DataFrame:
    """
    Generates and loads the portfolio investments' NAV time-series dataframe.

    :param product_details_csv_path: relative path of the product details csv file
    :param transactions_csv_path: relative path of the transactions csv file
    :param product_price_series_csv_path: relative path of the product price series csv file
    :param groupby_field: columns field by which to aggregate assets (it can happen that 2 assets have 2 different
    IDs but the same name -> in this case the NAVs would be aggregated)
    :param frequency: time-series frequency
    :param unprocessed_long_form: indicates whether to return he data in long form, with no change in frequency or
     mapping field
    :return: wide-form portfolio assets NAV dataframe
    """
    quotas_df = load_quotas_time_series(product_details_csv_path=product_details_csv_path,
                                        transactions_csv_path=transactions_csv_path,
                                        unprocessed_long_form=True)
    product_price_series = load_product_price_time_series(product_details_csv_path=product_details_csv_path,
                                                          product_price_series_csv_path=product_price_series_csv_path,
                                                          ffill_non_overlapping_bank_holidays=True,
                                                          unprocessed_long_form=True)
    nav_df = nav.generate_portfolio_asset_nav(quotas_df=quotas_df, price_time_series=product_price_series)

    if unprocessed_long_form:
        return nav_df

    # Change names to groupby_field
    product_id_to_field_dict = generate_product_id_to_field_dict(product_details_csv_path=product_details_csv_path,
                                                                 mapping_field=groupby_field)
    nav_df[groupby_field] = nav_df['product_id'].apply(lambda product_id: product_id_to_field_dict[product_id])
    # Aggregate by groupby_field
    nav_df = nav_df.groupby([groupby_field, 'date'])[['nav']].sum().reset_index(drop=False)

    nav_df = pd.pivot_table(nav_df, index='date', values='nav', columns=groupby_field)
    nav_df = nav_df.resample(frequency).last().dropna(how='all').replace(0, nan)
    return nav_df


def load_portfolio_nav(product_details_csv_path: str = PRODUCT_DETAILS_CSV_PATH,
                       transactions_csv_path: str = TRANSACTIONS_CSV_PATH,
                       frequency: str = '1D') -> pd.Series:
    """
    Generates and loads the portfolio' NAV time-series.

    :param product_details_csv_path: relative path of the product details csv file
    :param transactions_csv_path: relative path of the transactions csv file
    :param frequency: time-series frequency
    :return: portfolio's NAV time series
    """
    nav_df = load_portfolio_asset_nav(product_details_csv_path=product_details_csv_path,
                                      transactions_csv_path=transactions_csv_path,
                                      groupby_field='product_id',
                                      frequency=frequency,
                                      unprocessed_long_form=True)
    return nav.generate_portfolio_nav(nav_df)


def load_portfolio_asset_returns(product_details_csv_path: str = PRODUCT_DETAILS_CSV_PATH,
                                 transactions_csv_path: str = TRANSACTIONS_CSV_PATH,
                                 groupby_field='name',
                                 frequency: str = '1D',
                                 net: bool = True,
                                 unprocessed_long_form: bool = False) -> pd.DataFrame:
    """
    Generates and loads portfolio assets' returns time-series dataframe.

    :param product_details_csv_path: relative path of the product details csv file
    :param transactions_csv_path: relative path of the transactions csv file
    :param groupby_field: columns field by which to aggregate assets (it can happen that 2 assets have 2 different
    IDs but the same name -> in this case the NAVs would be aggregated)
    :param frequency: time-series frequency
    :param net: indicates whether the returns should be net of fees
    :param unprocessed_long_form: indicates whether to return he data in long form, with no change in frequency or
     mapping field
    :return: portfolio assets' returns time-series dataframe
    """
    # Transactions are needed for CF Calculations
    transactions_df = pd.read_csv(transactions_csv_path, parse_dates=['date'], index_col='id')
    nav_df = load_portfolio_asset_nav(unprocessed_long_form=True)
    cash_flows_df = nav.generate_cash_flows(transactions_df, net=net)
    # Change names to groupby_field
    product_id_to_field_dict = generate_product_id_to_field_dict(product_details_csv_path=product_details_csv_path,
                                                                 mapping_field=groupby_field)
    nav_df[groupby_field] = nav_df['product_id'].apply(lambda product_id: product_id_to_field_dict[product_id])
    cash_flows_df[groupby_field] = cash_flows_df['product_id'].\
        apply(lambda product_id: product_id_to_field_dict[product_id])

    # Aggregate by groupby_field
    nav_df = nav_df.groupby([groupby_field, 'date'])[['nav']].sum().reset_index(drop=False)
    cash_flows_df = cash_flows_df.groupby([groupby_field, 'date'])[['cf']].sum().reset_index(drop=False)

    # Compute Daily Returns
    daily_returns = nav.generate_portfolio_asset_nav_pnl_return_time_series(nav_df=nav_df,
                                                                            cash_flows_df=cash_flows_df,
                                                                            groupby_field=groupby_field)
    if unprocessed_long_form:
        return daily_returns
    # Long to wide
    daily_returns = pd.pivot_table(daily_returns, index='date', values='return', columns=groupby_field)
    # Resample
    returns = daily_returns.resample(frequency). \
        apply(lambda col: (1 + col).prod() - 1).replace(0, nan).dropna(how='all')
    return returns


def load_portfolio_returns(transactions_csv_path: str = TRANSACTIONS_CSV_PATH,
                           frequency: str = '1D',
                           net: bool = True,
                           unprocessed_long_form: bool = False) -> pd.DataFrame:
    """
    Generates and loads portfolio returns time-series.

    :param transactions_csv_path: relative path of the transactions csv file
    :param frequency: time-series frequency
    :param net: indicates whether the returns should be net of fees
    :param unprocessed_long_form: indicates whether to return he data in long form, with no change in frequency or
     mapping field
    :return: portfolio returns time-series
    """
    # Transactions are needed for CF Calculations
    transactions_df = pd.read_csv(transactions_csv_path, parse_dates=['date'], index_col='id')
    nav_df = load_portfolio_asset_nav(unprocessed_long_form=True)
    cash_flows_df = nav.generate_cash_flows(transactions_df, net=net)
    # Compute Daily Returns
    daily_returns = nav.generate_portfolio_nav_pnl_return_time_series(nav_df=nav_df,
                                                                      cash_flows_df=cash_flows_df)
    if unprocessed_long_form:
        return daily_returns

    # Long to wide
    daily_returns = daily_returns.set_index('date')
    daily_returns = daily_returns['portfolio_return']

    # Resample daily returns
    return daily_returns.resample(frequency). \
        apply(lambda col: (1 + col).prod() - 1).replace(0, nan).dropna(how='all').rename('portfolio_return')


def load_portfolio_asset_return_contributions(product_details_csv_path: str = PRODUCT_DETAILS_CSV_PATH,
                                              transactions_csv_path: str = TRANSACTIONS_CSV_PATH,
                                              groupby_field='name',
                                              frequency: str = '1D',
                                              net: bool = True,
                                              unprocessed_long_form: bool = False) -> pd.DataFrame:
    """
    Computes and load portfolio assets' return contributions.

    :param product_details_csv_path: relative path of the product details csv file
    :param transactions_csv_path: relative path of the transactions csv file
    :param groupby_field: columns field by which to aggregate assets (it can happen that 2 assets have 2 different
    IDs but the same name -> in this case the NAVs would be aggregated)
    :param frequency: time-series frequency
    :param net: indicates whether the returns should be net of fees
    :param unprocessed_long_form: indicates whether to return he data in long form, with no change in frequency or
     mapping field
    :return: portfolio assets' return contributions
    """
    # We need pnl data so unprocessed_long_form=True
    product_df = load_portfolio_asset_returns(transactions_csv_path=transactions_csv_path,
                                              groupby_field='product_id',
                                              net=net, unprocessed_long_form=True)
    portfolio_df = load_portfolio_returns(transactions_csv_path=transactions_csv_path,
                                          net=net, unprocessed_long_form=True)
    # Merge the data
    merged_df = product_df.merge(portfolio_df, on=['date'], how='inner')

    if unprocessed_long_form:
        return merged_df

    # Change Frequency and aggregate PNLs
    pnl_df = merged_df.set_index('date').groupby(['product_id']).\
        resample(frequency).agg({'pnl': 'sum', 'portfolio_pnl': 'sum'}).reset_index(drop=False)
    pnl_df['%pnl'] = pnl_df['pnl'] / pnl_df['portfolio_pnl']
    # Compute time-weighted portfolio returns
    portfolio_return_df = load_portfolio_returns(transactions_csv_path=transactions_csv_path,
                                                 net=net, frequency=frequency).reset_index(drop=False)
    # Merge
    pnl_merged_df = pnl_df.merge(portfolio_return_df, on=['date'], how='inner')
    # Compute Contribution
    pnl_merged_df['contribution'] = pnl_merged_df['%pnl']*pnl_merged_df['portfolio_return']

    # Rename products
    product_id_to_field_dict = generate_product_id_to_field_dict(product_details_csv_path=product_details_csv_path,
                                                                 mapping_field=groupby_field)
    pnl_merged_df[groupby_field] = pnl_merged_df['product_id'].apply(lambda product_id:
                                                                          product_id_to_field_dict[product_id])
    # Groupby (product with different IDs but same groupby_field)
    pnl_merged_df = pnl_merged_df.groupby([groupby_field, 'date'])[['contribution']].sum().reset_index(drop=False)

    # Long to Wide
    return pd.pivot_table(pnl_merged_df, index='date', columns=groupby_field, values='contribution')


def load_portfolio_asset_pnl(product_details_csv_path: str = PRODUCT_DETAILS_CSV_PATH,
                             transactions_csv_path: str = TRANSACTIONS_CSV_PATH,
                             groupby_field='name',
                             frequency: str = '1D',
                             net: bool = True,
                             unprocessed_long_form: bool = False) -> pd.DataFrame:
    """
    Computes and load portfolio assets' P&L.

    :param product_details_csv_path: relative path of the product details csv file
    :param transactions_csv_path: relative path of the transactions csv file
    :param groupby_field: columns field by which to aggregate assets (it can happen that 2 assets have 2 different
    IDs but the same name -> in this case the NAVs would be aggregated)
    :param frequency: time-series frequency
    :param net: indicates whether the returns should be net of fees
    :param unprocessed_long_form: indicates whether to return he data in long form, with no change in frequency or
     mapping field
    :return: portfolio assets' P&L
    """
    # We need pnl data so unprocessed_long_form=True
    portfolio_asset_df = load_portfolio_asset_returns(transactions_csv_path=transactions_csv_path,
                                                      groupby_field='product_id',
                                                      net=net, unprocessed_long_form=True)
    if unprocessed_long_form:
        return portfolio_asset_df

    # Change Frequency and aggregate PNLs
    pnl_df = portfolio_asset_df.set_index('date').groupby(['product_id']).\
        resample(frequency).agg({'pnl': 'sum'}).reset_index(drop=False)

    # Rename products
    product_id_to_field_dict = generate_product_id_to_field_dict(product_details_csv_path=product_details_csv_path,
                                                                 mapping_field=groupby_field)
    pnl_df[groupby_field] = pnl_df['product_id'].apply(lambda product_id: product_id_to_field_dict[product_id])
    # Groupby (product with different IDs but same groupby_field)
    pnl_merged_df = pnl_df.groupby([groupby_field, 'date'])[['pnl']].sum().reset_index(drop=False)

    return pd.pivot_table(pnl_merged_df, index='date', columns=groupby_field, values='pnl')


if __name__ == '__main__':
    pass
