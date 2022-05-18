from datetime import datetime
import itertools

from numpy import nan
import pandas as pd


def generate_quotas(transactions_df: pd.DataFrame) -> pd.DataFrame:
    # Date Range starting from the first transaction
    dates = pd.bdate_range(
        transactions_df['date'].min().date(), datetime.today().date())

    # Gather all product IDs
    product_id_ls = list(set(transactions_df['product_id']))
    product_id_ls.sort()

    # Initialize quotas df zeroes
    quotas_df = pd.DataFrame(columns=['product_id', 'date', 'quota'])
    quotas_df['product_id'] = list(itertools.chain(*[[item] * len(dates) for item in product_id_ls]))
    quotas_df['date'] = dates.to_list() * len(product_id_ls)
    quotas_df['quota'] = 0

    # Loop through each transaction and generate the quotas
    for _, row in transactions_df.iterrows():
        start_date = row['date']
        transaction_sign = 1 if row['buysell'] == 'B' else -1

        # Add/Remove instruments
        quotas_df.loc[(quotas_df['date'] >= start_date) & (quotas_df['product_id'] == row['product_id']), 'quota'] \
            += transaction_sign * row['quantity']

    # Sort by product_id, Date
    quotas_df = quotas_df.replace(0, nan).dropna()
    return quotas_df.sort_values(by=['product_id', 'date'], ascending=True).reset_index(drop=True)


def generate_cash_flows(transactions_df: pd.DataFrame, net=True) -> pd.DataFrame:
    # Date Range starting from the first transaction
    dates = pd.bdate_range(
        transactions_df['date'].min().date(), datetime.today().date())

    # Gather all product IDs
    product_id_ls = list(set(transactions_df['product_id']))
    product_id_ls.sort()

    # Initialize cf df zeroes
    cash_flows_df = pd.DataFrame(columns=['product_id', 'date', 'cf'])
    cash_flows_df['product_id'] = list(itertools.chain(*[[item] * len(dates) for item in product_id_ls]))
    cash_flows_df['date'] = dates.to_list() * len(product_id_ls)
    cash_flows_df['cf'] = 0

    # CF field to consider
    cash_flows_field = 'total_plus_fee_in_base_currency' if net else 'total_in_base_currency'

    # Loop through each transaction and generate the CF df
    for _, row in transactions_df.iterrows():
        flow_date = row['date']
        # Add cash flows
        cash_flows_df.loc[(cash_flows_df['date'] == flow_date) & (cash_flows_df['product_id'] == row['product_id']),
                          'cf'] += row[cash_flows_field]
    cash_flows_df = cash_flows_df.replace(0, nan).dropna()
    return cash_flows_df


def generate_fees(transactions_df: pd.DataFrame) -> pd.DataFrame:
    cash_flows_gross_df = generate_cash_flows(
        transactions_df, net=False)
    cash_flows_net_df = generate_cash_flows(
        transactions_df, net=True)
    fees_df = cash_flows_gross_df.copy(deep=True)
    fees_df['fee'] = fees_df['cf'] - cash_flows_net_df['cf']
    return fees_df[['product_id', 'date', 'cf', 'fee']]


def generate_portfolio_asset_nav(quotas_df: pd.DataFrame, price_time_series: pd.DataFrame) -> pd.DataFrame:
    # Generate NAV time series from 2 long form seres
    merged_df = price_time_series.merge(quotas_df, on=['date', 'product_id'], how='inner')
    merged_df['nav'] = merged_df['price'] * merged_df['quota']
    merged_df['nav'] = merged_df['nav'].replace(0, nan)
    return merged_df.loc[~merged_df['nav'].isna()]


def generate_portfolio_nav(nav_df: pd.DataFrame) -> pd.Series:
    # Generate Portfolio NAV time series
    nav_df = nav_df.copy(deep=True)

    # Long -> Wide -> Long to treat nans in the middle of the series
    nav_df = pd.pivot_table(nav_df, index='date', values='nav', columns='product_id').replace(0, nan).ffill()
    nav_df = nav_df.melt(ignore_index=False, var_name='product_id', value_name='portfolio_nav').\
        reset_index(drop=False).dropna()

    nav_df = nav_df.groupby('date').agg({'portfolio_nav': 'sum'})
    nav_df['portfolio_nav'] = nav_df['portfolio_nav'].replace(0, nan).dropna()
    return nav_df


def generate_portfolio_cash_flow(cash_flows_df: pd.DataFrame) -> pd.Series:
    # Generate Portfolio NAV time series
    cash_flows_df = cash_flows_df.copy(deep=True)
    cash_flows_df['cf'] = cash_flows_df['cf'].fillna(0)
    return cash_flows_df.groupby('date').agg({'cf': 'sum'}).rename({'cf': 'portfolio_cf'}, axis=1).\
        reset_index(drop=False)


def generate_returns(df: pd.DataFrame, groupby_field: str = None):
    # Compute beginning of period nav
    if groupby_field is not None:
        prefix = ''
        df[prefix + 'begin_nav'] = df.groupby([groupby_field])[[prefix + 'nav']].shift(1).fillna(0)
    else:
        prefix = 'portfolio_'
        df[prefix + 'begin_nav'] = df[[prefix + 'nav']].shift(1).fillna(0)
    # Compute Returns
    # PnL_t = NAV_t - (NAV_{t-1} + CF_{t-1})
    # R_t = PnL_t/(NAV_{t-1} + CF_{t-1})
    # Cash Flow will be subtracted in code since the formula above considers cash to investment = - (cash to investor)
    df[prefix + 'pnl'] = df[prefix + 'nav'] - (df[prefix + 'begin_nav'] - df[prefix + 'cf'])
    df[prefix + 'return'] = df[prefix + 'pnl']/(df[prefix + 'begin_nav'] - df[prefix + 'cf'])
    return df.dropna(how='all')


def generate_portfolio_asset_nav_pnl_return_time_series(nav_df: pd.DataFrame,
                                                        cash_flows_df: pd.DataFrame,
                                                        groupby_field: str = 'product_id'
                                                        ) -> pd.DataFrame:
    # Merge Nav and Cash Flows
    cash_flows_df['cf'] = cash_flows_df['cf'].replace(0, nan)
    merged_df = nav_df.merge(cash_flows_df.dropna(), on=['date', groupby_field], how='outer')
    # Adjustments
    merged_df['nav'] = merged_df.groupby([groupby_field])[['nav']].ffill()
    merged_df['cf'] = merged_df['cf'].fillna(0)

    # Sort dataframe
    merged_df = merged_df.sort_values(by=[groupby_field, 'date'])

    return generate_returns(merged_df, groupby_field=groupby_field)


def generate_portfolio_nav_pnl_return_time_series(nav_df: pd.DataFrame,
                                                  cash_flows_df: pd.DataFrame
                                                  ) -> pd.Series:
    # generate portfolio NAV
    nav_df = generate_portfolio_nav(nav_df=nav_df)

    # Generate portfolio Cash Flows
    cash_flows_df = generate_portfolio_cash_flow(cash_flows_df=cash_flows_df)

    # Merge Nav and Cash Flows
    prefix = 'portfolio_'
    cash_flows_df[prefix + 'cf'] = cash_flows_df[prefix + 'cf'].replace(0, nan)
    merged_df = nav_df.merge(cash_flows_df.dropna(), on=['date'], how='outer')
    # Adjustments
    merged_df[prefix + 'nav'] = merged_df[[prefix + 'nav']].ffill()
    merged_df[prefix + 'cf'] = merged_df[prefix + 'cf'].fillna(0)

    return generate_returns(merged_df, groupby_field=None)


if __name__ == '__main__':
    pass
