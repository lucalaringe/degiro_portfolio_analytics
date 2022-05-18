import calendar
from datetime import datetime

import scipy.optimize
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

ROLLING_FACTOR = 36


def performance_stats(return_series: pd.Series, risk_free_rate: pd.Series = None, periods_per_year=12,
                      styled: bool = False):
    """Takes a time-series of returns and returns a series of performance statistics.

    :param return_series: time-series of returns
    :type return_series: pd.Series
    :param risk_free_rate: time-series of risk-free rate asset returns (e.g. libor)
    :type risk_free_rate: pd.Series
    :param periods_per_year: inverse yearly frequency of observations, defaults to 12
    :type periods_per_year: int, optional
    :param styled: indicated whether to output the styled table or the simple dataframe, defaults to False
    :type styled: bool, optional    
    :return: dataframe of returns' statistics
    :rtype: pd.io.formats.style.Styler
    """
    # Data wrangling
    return_series = pd.DataFrame(return_series).copy(deep=True)
    return_series_name = return_series.columns
    return_series.dropna(how='all', inplace=True)

    if risk_free_rate is not None:
        risk_free_rate_name = risk_free_rate.columns if len(
            risk_free_rate.shape) > 1 else [risk_free_rate.name]
        joined_data = pd.DataFrame(return_series).join(
            risk_free_rate, how='left')
        risk_free_rate = joined_data[risk_free_rate_name]
        return_series = joined_data[return_series_name]
        if risk_free_rate.isna().sum().squeeze() > 0:
            print('Warning! Not enough Risk Free returns data points. Considering using a different time frame.')
            return None

    def func_2(series):
        return annualize_returns(series, periods_per_year)

    func_2.__name__ = "Return (Ann.)"

    def func_3(series):
        return annualize_vol(series, periods_per_year)

    func_3.__name__ = "Volatility (Ann.)"

    def func_4(series):
        return skewness(series)

    func_4.__name__ = "Skewness"

    def func_5(series):
        return kurtosis(series)

    func_5.__name__ = "Kurtosis"

    def func_6(series):
        if risk_free_rate is None:
            return pd.Series([np.nan], index=series.name)
        return sharpe_ratio(series, risk_free_rate, periods_per_year)

    func_6.__name__ = "Sharpe Ratio"

    def func_7(series):
        return var_gaussian(series, modified=True)

    func_7.__name__ = "VaR 95%"

    def func_8(series):
        return cvar_historic(series)

    func_8.__name__ = "CVaR 95%"

    def func_9(series):
        return max_drawdown(series)

    func_9.__name__ = "Max Drawdown"

    def func_10(series):
        return sum(series > 0) / len(series.dropna())

    func_10.__name__ = "Up Months"

    def func_11(series):
        return sum(series < 0) / len(series.dropna())

    func_11.__name__ = "Down Months"

    def func_12(series):
        return series.max()

    func_12.__name__ = "Largest Positive Month"

    def func_13(series):
        return series.min()

    func_13.__name__ = "Largest Negative Month"

    def func_15(series):
        return longest_drawdown(series)

    func_15.__name__ = "Longest Drawdown(Months)"

    result = return_series. \
        apply([func_2, func_3, func_4, func_5, func_6, func_7, func_8, func_9, func_15, func_10, func_11, func_12,
               func_13], axis=0).T

    # Styling
    if styled:
        col_styles = {}
        for col in result.columns:
            if col in ['Sharpe Ratio', 'Skewness', 'Kurtosis']:
                col_styles[col] = '{:,.2f}'
            elif col in ['Sample Size', func_15.__name__]:
                col_styles[col] = '{:,.0f}'
            else:
                col_styles[col] = '{:,.2%}'
        result = result.style.format(col_styles, na_rep="-")

    return result


def return_stats(return_series: pd.Series, risk_free_rate: pd.Series = None, periods_per_year: int = 12,
                 styled: bool = False):
    """Takes a time-series of returns and returns a series of returns' statistics.

    :param return_series: time-series of returns
    :type return_series: pd.Series
    :param risk_free_rate: time-series of risk-free rate asset returns (e.g. libor)
    :type risk_free_rate: pd.Series
    :param periods_per_year: inverse yearly frequency of observations, defaults to 12
    :param styled: indicated whether to output the styled table or the simple dataframe, defaults to False
    :type styled: bool, optional    
    :return: dataframe of returns' statistics
    :rtype: pd.io.formats.style.Styler
    """
    # Data wrangling
    return_series = pd.DataFrame(return_series).copy(deep=True)
    return_series_name = return_series.columns
    return_series.dropna(how='all', inplace=True)

    if risk_free_rate is not None:
        risk_free_rate_name = risk_free_rate.columns if len(
            risk_free_rate.shape) > 1 else [risk_free_rate.name]
        joined_data = pd.DataFrame(return_series).join(
            risk_free_rate, how='inner')
        risk_free_rate = joined_data[risk_free_rate_name]
        return_series = joined_data[return_series_name]
        if risk_free_rate.isna().sum().squeeze() > 0:
            print('Warning! Not enough Risk Free returns data points. Considering using a different time frame.')
            return None

    # Current year and quarter
    year = return_series.index[-1].year
    quarter = return_series.index[-1].quarter

    def func_0(series):
        return series.iloc[-1]

    func_0.__name__ = 'MTD Returns'

    def func_1(series):
        return (
                (1 + series[(series.index.quarter == quarter) & (series.index.year == year)]).prod() - 1)

    func_1.__name__ = 'QTD Returns'

    def func_2(series):
        return (1 + series[series.index.year == year]).prod() - 1

    func_2.__name__ = 'YTD Returns'

    def func_3(series):
        # Not enough data
        if series.dropna().shape[0] < 12 or series.name == 'Libor 1M Monthly Return':
            return pd.Series([np.nan], index=series.name)
        return annualize_returns(series.tail(12), periods_per_year)

    func_3.__name__ = 'L12M Returns'

    def func_4(series):
        # Not enough data
        if series.dropna().shape[0] < 12 or series.name == 'Libor 1M Monthly Return':
            return pd.Series([np.nan], index=series.name)
        return annualize_vol(series.tail(12), periods_per_year)

    func_4.__name__ = 'L12M Volatility'

    def func_5(series):
        # Not enough data
        if series.dropna().shape[0] < 12 or series.name == 'Libor 1M Monthly Return':
            return pd.Series([np.nan], index=series.name)
        elif risk_free_rate is None:
            return pd.Series([np.nan], index=series.name)
        return sharpe_ratio(series.tail(12), risk_free_rate.tail(12), periods_per_year)

    func_5.__name__ = 'L12M Sharpe'

    def func_6(series):
        # Not enough data
        if series.dropna().shape[0] < 36 or series.name == 'Libor 1M Monthly Return':
            return pd.Series([np.nan], index=series.name)
        return annualize_returns(series.tail(36), periods_per_year)

    func_6.__name__ = 'L3Y Returns'

    def func_7(series):
        # Not enough data
        if series.dropna().shape[0] < 36 or series.name == 'Libor 1M Monthly Return':
            return pd.Series([np.nan], index=series.name)
        return annualize_vol(series.tail(36), periods_per_year)

    func_7.__name__ = 'L3Y Volatility'

    def func_8(series):
        # Not enough data
        if series.dropna().shape[0] < 36 or series.name == 'Libor 1M Monthly Return':
            return pd.Series([np.nan], index=series.name)
        elif risk_free_rate is None:
            return pd.Series([np.nan], index=series.name)
        return sharpe_ratio(series.tail(36), risk_free_rate.tail(36), periods_per_year)

    func_8.__name__ = 'L3Y Sharpe'

    def func_9(series):
        return annualize_returns(series, periods_per_year)

    func_9.__name__ = 'ITD Returns'

    def func_10(series):
        return annualize_vol(series, periods_per_year)

    func_10.__name__ = 'ITD Volatility'

    def func_11(series):
        if risk_free_rate is None:
            return pd.Series([np.nan], index=series.name)
        return sharpe_ratio(series, risk_free_rate, periods_per_year)

    func_11.__name__ = 'ITD Sharpe'

    result = return_series.apply([func_0, func_1, func_2, func_3, func_4, func_5, func_6, func_7,
                                  func_8, func_9, func_10, func_11], axis=0).T

    # Styling
    if styled:
        col_styles = {}
        for col in result.columns:
            col_styles[col] = '{:,.2f}' if 'Sharpe' in col else '{:,.2%}'
        result = result.style.format(col_styles, na_rep="-")

    return result


def xnpv(cash_flow_series: pd.Series, discount_rate: float) -> np.float64:
    """Computes NPV as the XNPV function in excel. The input is a date-indexed Cash Flow Series and a discount rate.
    Note: the values are discounted at the minimum date there is in the series, which in most of the cases corresponds
    to an initial cash outflow.

    :param cash_flow_series: date indexed series of cash flows
    :type cash_flow_series: pd.Series
    :param discount_rate: discount rate for the NPV calculations
    :type discount_rate: float
    :return: NPV as of the first date of the series
    :rtype: np.float64
    """
    dates = cash_flow_series.index
    values = cash_flow_series.values
    if discount_rate <= -1.0:
        return float('inf')
    d0 = min(dates)  # Date at which the values are discounted
    return sum([vi / (1.0 + discount_rate) ** ((di - d0).days / 365.0) for vi, di in zip(values, dates)])


def xirr(cash_flow_series: pd.Series):
    """Computes IRR as the XNPV function in excel. The input is a date-indexed Cash Flow Series.
    The function optimizes the discount rate such that the NPV is at 0. 
    Note: as for the NPV calculation, the values are discounted at the minimum date there is in the series,
    which in most of the cases corresponds to an initial cash outflow.

    :param cash_flow_series: date indexed series of cash flows
    :type cash_flow_series: pd.Series
    :return: IRR
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    cash_flow_series = pd.DataFrame(cash_flow_series).copy(deep=True)
    n_series = cash_flow_series.shape[1]

    if n_series > 1:  # Recursive case to deal with multiple series
        return cash_flow_series.dropna(how='all').apply(lambda series: xirr(series), axis=0)

    try:
        return scipy.optimize.newton(lambda discount_rate: xnpv(cash_flow_series.iloc[:, 0], discount_rate), 0.0)
    except RuntimeError:  # Failed to converge?
        return scipy.optimize.brentq(lambda discount_rate: xnpv(cash_flow_series.iloc[:, 0], discount_rate), -1.0, 1e10)


def excess_return(return_series: pd.Series, benchmark_series: pd.Series, annualized: bool = True,
                  periods_per_year: int = 12, dropna: bool = True) -> np.float64:
    """Computes (annualized) excess return of a series vs its benchmark.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param benchmark_series: non cumulative benchmark return series
    :type benchmark_series: pd.Series
    :param annualized: indicates whether to use annualized returns vs. total returns, defaults to True
    :type annualized: bool, optional
    :param periods_per_year: inverse yearly frequency of observations, defaults to 12
    :type periods_per_year: int, optional    
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional        
    :return: (annualized) excess return
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    if n_series > 1:  # Recursive case to deal with multiple series
        return return_series.dropna(how='all'). \
            apply(lambda series: excess_return(series, benchmark_series=benchmark_series, annualized=annualized,
                                               periods_per_year=periods_per_year, dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)

    joined_data = return_series.join(benchmark_series, how='left')
    return_series = joined_data.iloc[:, :1]
    benchmark_series = joined_data.iloc[:, 1]
    # Check there are no nans in the benchmark series
    if benchmark_series.isna().sum().squeeze() > 0:
        print('Warning! Not enough Benchmark returns data points.')
        return np.nan

    # Compute Excess Returns
    if annualized:
        return annualize_returns(return_series, periods_per_year=periods_per_year) - \
               annualize_returns(benchmark_series, periods_per_year=periods_per_year)
    return total_return(return_series) - total_return(benchmark_series)


def univariate_regression(y_series, x_dfs, libor=None):
    y_series = y_series.dropna()
    joined_data = pd.DataFrame(y_series).join(x_dfs, how='inner')
    # Needed if x_dfs series series is longer than y series
    x_dfs = joined_data[x_dfs.columns]
    # Needed if y series is longer than x_dfs series
    y_series = joined_data[y_series.name]
    if libor is not None:
        libor = pd.DataFrame(y_series).join(libor)[libor.name]
        x_dfs = x_dfs.sub(libor, axis=0)
        y_series = y_series - libor
    result = pd.DataFrame(index=x_dfs.columns, columns=[
        'Beta', 'std err', 't-stat', 'p-value', 'Correlation', 'Alpha'])
    for _, col in enumerate(x_dfs.columns):
        x = x_dfs[col]
        x = sm.add_constant(x)
        y = y_series
        model = sm.OLS(y, x).fit()
        df = pd.read_html(
            model.summary().tables[1].as_html(), header=0, index_col=0)[0]
        df = df.rename(
            columns={'coef': 'Beta', 't': 't-stat', 'P>|t|': 'p-value'})
        result.loc[df.index[1]] = df.iloc[1, :4]
        result.loc[df.index[1], 'Correlation'] = y.corr(x_dfs[col])
        beta = result.loc[df.index[1], 'Beta']
        result.loc[df.index[1], 'Alpha'] = _aggregated_monthly_returns(
            y) - beta * _aggregated_monthly_returns(x_dfs[col])
    return result


def regression(return_series: pd.Series, factors: pd.DataFrame, risk_free_rate: pd.Series = None,
               periods_per_year=12, annualize_alpha=False):
    """
    Runs a static single or multiple factor model on the provided data. The aim is to attribute the total return of an
    investment to
    various factors and to alpha.

    :param return_series: series of returns from an investment to be analyzed
    :type return_series: pd.Series
    :param factors: DataFrame of factor returns
    :type factors: pd.DataFrame
    :param risk_free_rate: time-series of risk-free rate asset returns (e.g. libor). It is subtracted to return_series
    if not None, defaults to None
    :type risk_free_rate: pd.Series, optional
    :param periods_per_year: yearly frequency of observations
    :type periods_per_year: int
    :param annualize_alpha: Indicates whether the alpha should be annualized
    :type annualize_alpha: bool
    :return: DataFrame containing regression results
    :rtype: pd.DataFrame
    """
    factors = pd.DataFrame(factors)
    factors_names = factors.columns

    joined_data = pd.DataFrame(return_series).join(
        factors, how='inner').dropna()
    y = joined_data.iloc[:, 0]
    factors = joined_data[factors_names]

    if risk_free_rate is not None:
        risk_free_rate = pd.DataFrame(risk_free_rate)
        risk_free_rate = pd.DataFrame(y).join(risk_free_rate)[
            risk_free_rate.columns[0]]
        # Risk Free Rate is subtracted to fund returns by default
        y = y.sub(risk_free_rate)

    factors = sm.add_constant(factors)

    # Static Regression
    model = sm.OLS(y, factors).fit()
    df = pd.read_html(
        model.summary().tables[1].as_html(), header=0, index_col=0)[0]
    df = df.iloc[:, :4].T.astype(float)

    if annualize_alpha:
        df.loc['coef', 'const'] = (
                                          1 + df.loc['coef', 'const']) ** periods_per_year - 1  # Annualize Alpha
        # Update std err using Delta method
        df.loc['std err', 'const'] *= periods_per_year * \
                                      (df.loc['coef', 'const'] + 1) ** (periods_per_year - 1)
        df.columns = ['Annualized Alpha'] + list(factors_names)
    else:
        df.columns = ['Alpha'] + list(factors_names)  # Not annualized

    df.rename(index={'t': 't-stat', 'P>|t|': 'p-value'}, inplace=True)
    return df.T.round({'coef': 4, 'std err': 3})


def rolling_regression(return_series: pd.Series, factors: pd.DataFrame, risk_free_rate: pd.Series = None,
                       rolling_factor=36, periods_per_year=12, confidence_intervals=False, confidence_level=0.05,
                       annualize_alpha=False):
    """
    Runs a rolling single or multiple factor model on the provided data. The aim is to attribute the total return of an
    investment to
    various factors and to alpha.

    :param return_series: series of returns from an investment to be analyzed
    :type return_series: pd.Series
    :param factors: DataFrame of factor returns
    :type factors: pd.DataFrame
    :param risk_free_rate: time-series of risk-free rate asset returns (e.g. libor). It is subtracted to return_series
    if not None, defaults to None
    :type risk_free_rate: pd.Series, optional
    :param rolling_factor: rolling factor to be applied to the rolling regression, defaults to 36
    :type rolling_factor: int, optional
    :param periods_per_year: yearly frequency of observations
    :type periods_per_year: int
    :param confidence_intervals: Indicates whether to also output the confidence intervals dataframe, defaults to False
    :type confidence_intervals: bool, optional
    :param confidence_level: Confidence level used to compute confidence intervals
    :type confidence_level: float
    :param annualize_alpha: Indicates whether the alpha should be annualized
    :type annualize_alpha: bool
    :return: DataFrame containing regression coefficients (or tuple of coefficients df and CI df)
    :rtype: pd.DataFrame
    """
    factors = pd.DataFrame(factors)
    factors_names = factors.columns

    joined_data = pd.DataFrame(return_series).join(
        factors, how='inner').dropna()
    y = joined_data.iloc[:, 0]
    factors = joined_data[factors_names]

    if risk_free_rate is not None:
        risk_free_rate = pd.DataFrame(risk_free_rate)
        risk_free_rate = pd.DataFrame(y).join(risk_free_rate)[
            risk_free_rate.columns[0]]
        # Risk Free Rate is  subtracted to fund returns by default
        y = y.sub(risk_free_rate)

    factors = sm.add_constant(factors)

    # Rolling Regression
    model = RollingOLS(y, factors, window=rolling_factor)
    model_fit = model.fit()
    # Coefficients
    coef_df = model_fit.params.copy()
    alpha_label = 'Annualized Alpha' if annualize_alpha else 'Alpha'

    if annualize_alpha:
        coef_df['const'] = (1 + coef_df['const']
                            ) ** periods_per_year - 1  # Annualize Alpha
    # Renaming
    coef_df.columns = [alpha_label] + list(coef_df.columns[1:])

    if confidence_intervals:
        conf_int_df = model_fit.conf_int(confidence_level).copy()
        if annualize_alpha:
            conf_int_df['const'] = (
                                           1 + conf_int_df['const']) ** periods_per_year - 1  # Annualize Alpha
        # Renaming
        conf_int_df.columns = pd.MultiIndex.from_tuples(
            [(alpha_label, 'lower'), (alpha_label, 'upper')] + list(conf_int_df.columns[2:]))
        return coef_df.dropna(), conf_int_df.dropna()

    return coef_df.dropna()


def top_drawdowns_table(return_series: pd.Series, n: int = None):
    """ Computes top n drawdowns and outputs a table with the following fields: Net Drawdown(%) (index), Peak Date,
     Valley Date, Recovery Date, Duration(Months).

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param n: number of top drawdowns, if None gets all drawdowns, defaults to None
    :type n: int, optional    
    :return: top n drawdowns table
    :rtype: pd.DataFrame
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True).dropna()

    # Cumulative returns
    cum_returns = (1 + return_series).cumprod()
    # Drawdowns
    drawdowns = cum_returns / cum_returns.cummax() - 1
    drawdowns.columns = ['Drawdown']
    dd = drawdowns['Drawdown']

    # Drawdown Cycles Table
    drawdown_cycles = _drawdown_cycles_table(
        return_series, drawdown_threshold=0)

    if drawdown_cycles.shape[0] == 0:  # No Drawdowns Detected
        return pd.DataFrame(columns=['Peak Date', 'Valley Date', 'Recovery Date', 'Duration(Months)'])

    drawdown_cycles.drop('end_recovery', axis=1, inplace=True)
    drawdown_cycles['dd'] = (-drawdown_cycles['dd'] * 100).round(2)

    # Compute End of Drawdown (different fon end_recovery)
    end_drawdown = dd[(dd == 0) & (dd.shift(1) != dd)
                      ].index.strftime('%Y-%m-%d')
    try:
        drawdown_cycles['Recovery Date'] = end_drawdown[1:]
    except ValueError:  # Drawdown is continuing today
        drawdown_cycles['Recovery Date'] = end_drawdown[1:].tolist(
        ) + [return_series.index.strftime('%Y-%m-%d')[-1]]
    drawdown_cycles[['start_dd', 'start_recovery']] = drawdown_cycles[[
        'start_dd', 'start_recovery']].applymap(lambda x: x.strftime('%Y-%m-%d'))
    drawdown_cycles = drawdown_cycles.rename(
        columns={'start_dd': 'Peak Date', 'start_recovery': 'Valley Date', 'dd': 'Net drawdown(%)'})

    # Sort
    drawdown_cycles.sort_values(
        by='Net drawdown(%)', ascending=False, inplace=True)
    drawdown_cycles.set_index('Net drawdown(%)', drop=True, inplace=True)

    # Add duration
    drawdown_cycles['Duration(Months)'] = drawdown_cycles.apply(lambda row: round(
        (datetime.strptime(row['Recovery Date'], r"%Y-%m-%d") -
         datetime.strptime(row['Peak Date'], r"%Y-%m-%d")).days / 30), axis=1)

    if n is not None:
        n_drawdowns = drawdown_cycles.shape[0]
        if n > n_drawdowns:
            print(
                f'Warning! n ({n}) is greater than the number of drawdowns detected ({n_drawdowns}): '
                f'n will be lowered to {n_drawdowns} and all drawdowns will be shown.')
            n = n_drawdowns
        drawdown_cycles = drawdown_cycles.iloc[:n, :]

    return drawdown_cycles


def _drawdown_cycles_table(return_series: pd.Series, drawdown_threshold=-0.01):
    """ Helper function that computes drawdowns below a certain thresholds and the relevant dates associated with them.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param drawdown_threshold: threshold for a drawdown to be considered as such (must be lower than 0),
    defaults to -0.01
    :type drawdown_threshold: float, optional       
    :return: drawdown cycles table table
    :rtype: pd.DataFrame
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True).dropna()

    # Cumulative returns
    cum_returns = (1 + return_series).cumprod()
    # Drawdowns
    drawdowns = cum_returns / cum_returns.cummax() - 1
    drawdowns.columns = ['Drawdown']

    dd = drawdowns['Drawdown']

    # Find drawdown cycles
    zero_dd_dates = dd[(dd == 0) & (dd.shift(-1) != dd)].index.tolist()
    zero_dd_dates.append(dd.index[-1])

    dd_cycles = pd.DataFrame(dtype=np.float64, columns=[
        'start_dd', 'start_recovery', 'end_recovery', 'dd'])
    for i in range(len(zero_dd_dates) - 1):
        t0, t1 = zero_dd_dates[i:i + 2]
        dd_slice = dd[(t0 <= dd.index) & (dd.index <= t1)]
        if dd_slice.min() < drawdown_threshold:
            idx = dd_slice.idxmin()
            dd_cycles.loc[t0, 'start_dd'] = t0
            dd_cycles.loc[t0, 'start_recovery'] = idx
            dd_cycles.loc[t0, 'dd'] = dd_slice.min()

    end_recovery = dd_cycles['start_dd'].shift(-1)
    try:
        end_recovery[-1] = zero_dd_dates[-1] if zero_dd_dates[-1] > dd_cycles['start_dd'][-1] else dd.index[-1]
    except IndexError:
        print(
            f'Warning! No drawdowns detected at the {drawdown_threshold} threshold!')
    dd_cycles['end_recovery'] = end_recovery

    return dd_cycles


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series.
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()
    return exp / sigma_r ** 3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series.
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 4).mean()
    return exp / sigma_r ** 4


def annualize_returns(return_series: pd.Series, periods_per_year: int = 12, dropna: bool = True):
    """ Annualizes a set of returns.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param periods_per_year: inverse yearly frequency of observation, defaults to 12: monthly
    :type periods_per_year: int, optional
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional    
    :return: annualized returns
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]
    n_periods = return_series.shape[0]

    if n_series > 1:  # Recursive Case
        return return_series.dropna(how='all'). \
            apply(lambda col: annualize_returns(col, periods_per_year=periods_per_year, dropna=dropna), axis=0)

    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)

    compounded_growth = (1 + return_series).prod().squeeze()
    annualized_returns = np.math.pow(
        compounded_growth, (periods_per_year / n_periods)) - 1
    return annualized_returns  # Base Case


def _aggregated_monthly_returns(r):
    """
    Compounded returns - annualized if number of periods is more than 12, else not annualized.
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return (compounded_growth - 1) if n_periods <= 12 else compounded_growth ** (12 / n_periods) - 1


def annualize_vol(return_series: pd.Series, periods_per_year: int = 12, dropna: bool = True):
    """ Annualizes the vol of a set of returns.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param periods_per_year: inverse yearly frequency of observation, defaults to 12: monthly
    :type periods_per_year: int, optional
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional        
    :return: annualized volatility
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    if n_series > 1:  # Recursive Case
        return return_series.dropna(how='all').apply(lambda col: annualize_vol(col, periods_per_year=periods_per_year,
                                                                               dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)
    annualized_vol = return_series.std() * (np.math.sqrt(periods_per_year))
    return annualized_vol.squeeze()


def sharpe_ratio(return_series: pd.Series, riskfree_rate: pd.Series, periods_per_year: int = 12, dropna: bool = True):
    """Computes the annualized sharpe ratio of a set of returns.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param riskfree_rate: risk free rate return series
    :type riskfree_rate: pd.Series
    :param periods_per_year: inverse yearly frequency of observation, defaults to 12: monthly
    :type periods_per_year: int, optional
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional    
    :return: sharpe ratio
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    # Recursive Case
    if n_series > 1:
        return return_series.dropna(how='all').apply(lambda series: sharpe_ratio(series, riskfree_rate=riskfree_rate,
                                                                                 dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)
    joined_data = return_series.join(riskfree_rate, how='inner')
    return_series = joined_data.iloc[:, :1]
    riskfree_rate = joined_data.iloc[:, 1]
    if riskfree_rate.isna().sum().squeeze() > 0:
        print('Warning! Not enough Risk Free returns data points.')
        return np.nan

    # Computation
    excess_return_series = pd.DataFrame(return_series).sub(
        riskfree_rate.values, axis=0)
    sharpe_ratio_float = excess_return_series.mean() / excess_return_series.std()
    annualized_sharpe_ratio = np.math.sqrt(periods_per_year) * sharpe_ratio_float

    return annualized_sharpe_ratio.squeeze()


def is_normal(return_series: pd.Series, level=0.01) -> bool:
    """Applies the Jarque-Bera test to determine if a Series is normal or not Test is applied at the 1% 
    level by default. Returns True if the hypothesis of normality not rejected, False otherwise.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param level: level of confidence of the test, defaults to 0.01
    :type level: float, optional
    :return: True if the hypothesis of normality not rejected, False otherwise.
    :rtype: bool
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    # Recursive Case
    if n_series > 1:
        return return_series.apply(lambda series: is_normal(series, level=level), axis=0)

    # Base Case
    _, p_value = scipy.stats.jarque_bera(return_series.dropna())
    return p_value > level


def var_historic(return_series: pd.Series, level: float = 0.05, dropna: bool = True) -> np.float64:
    """Returns the historic Value at Risk at a specified level i.e. returns the number such that "level"
    of the returns fall below that number, and the (100% - level) percent are above.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param level: VaR percentile level, must be in (0,1), defaults to 0.05
    :type level: float, optional
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional      
    :return: Historical VaR
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    # Recursive Case
    if n_series > 1:
        return return_series.dropna(how='all').apply(lambda series: var_historic(series, level=level, dropna=dropna),
                                                     axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return -np.percentile(return_series.dropna(), level * 100)


def cvar_historic(return_series: pd.Series, level: float = 0.05, dropna: bool = True) -> np.float64:
    """Computes the Conditional VaR of Series or DataFrame.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param level: VaR percentile level, must be in (0,1), defaults to 0.05
    :type level: float, optional
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional      
    :return: Historical CVaR
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    # Recursive Case
    if n_series > 1:
        return return_series.dropna(how='all').apply(lambda series: cvar_historic(series, level=level, dropna=dropna),
                                                     axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)
    conditional_observation_idx = return_series <= -var_historic(return_series, level=level, dropna=dropna)
    return -return_series.dropna()[conditional_observation_idx].mean().squeeze()


def var_gaussian(return_series: pd.Series, level: float = 5, modified: bool = False, dropna: bool = True) -> np.float64:
    """Returns the Parametric Gaussian VaR of a Series or DataFrame. If "modified" is True,
    then the modified VaR is returned, using the Cornish-Fisher modification.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param level: VaR percentile level, must be in (0,1), defaults to 0.05
    :type level: float, optional
    :param modified: indicates whether to use the Cornish-Fisher modification, defaults to False
    :type modified: bool, optional
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional      
    :return: Gaussian VaR
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    # Recursive Case
    if n_series > 1:
        return return_series.dropna(how='all').apply(lambda series: var_gaussian(series, level=level, modified=modified,
                                                                                 dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)
    # compute the Z score assuming it was Gaussian
    z = scipy.stats.norm.ppf(level / 100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(return_series)
        k = kurtosis(return_series)
        z = (z +
             (z ** 2 - 1) * s / 6 +
             (z ** 3 - 3 * z) * (k - 3) / 24 -
             (2 * z ** 3 - 5 * z) * (s ** 2) / 36
             )
    return -(return_series.dropna().mean() + z * return_series.dropna().std(ddof=1)).squeeze()


def max_drawdown(return_series: pd.Series, dropna: bool = True):
    """Computes the max drawdown for a given returns stream.

    :param return_series: fund non-cumulative return series
    :type return_series: pd.Series
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional        
    :return: max drawdown of the return series
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    # Recursive Case
    if n_series > 1:
        return return_series.dropna(how='all').apply(lambda series: max_drawdown(series, dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)

    cum_returns = (1 + return_series).cumprod()
    drawdowns = -(cum_returns / cum_returns.cummax() - 1)
    return drawdowns.max(axis=0).squeeze()


def longest_drawdown(return_series: pd.Series, dropna: bool = True):
    """Computes the n. of months of the longest drawdown for a given (monthly) returns stream.

    :param return_series: fund non-cumulative return series
    :type return_series: pd.Series
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional        
    :return: n. of months of the longest drawdown
    :rtype: int
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    # Recursive Case
    if n_series > 1:
        return return_series.dropna(how='all').apply(lambda series: longest_drawdown(series, dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)
    return top_drawdowns_table(return_series)['Duration(Months)'].max()


def bias_ratio(return_series: pd.Series, dropna: bool = True) -> np.float64:
    """Computes the bias ratio for a given returns stream. The bias ratio 
    can be used an indicator of the likelihood of return smoothing and illiquid assets NAV manipulation

    :param return_series: fund non-cumulative return series
    :type return_series: pd.Series
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional        
    :return: bias ratio of the return series
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    # Recursive Case
    if n_series > 1:
        return return_series.dropna(how='all').apply(lambda series: bias_ratio(series, dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series = return_series.dropna().iloc[:, 0]
    numerator = return_series[(return_series >= 0) & (return_series < return_series.std())].count()
    denominator = return_series[(return_series < 0) & (return_series >= -return_series.std())].count()
    return (numerator / denominator).squeeze()


def autocorrelation(return_series: pd.Series, lag=1, dropna: bool = True):
    """Computes autocorrelation of a given series at a specified lag.

    :param return_series: non-cumulative return series
    :type return_series: pd.Series
    :param lag: autocorrelation lag, defaults to 1
    :type lag: int, optional
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional        
    :return: autocorrelation of the return series
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    # Recursive Case
    if n_series > 1:
        return return_series.dropna(how='all').apply(lambda series: autocorrelation(series, lag=lag, dropna=dropna),
                                                     axis=0)
    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series = return_series.dropna()
    return return_series.corrwith(return_series.shift(lag)).values[0]


def hit_ratio(return_series: pd.Series, target_series: pd.Series = None, dropna: bool = True):
    """Computes hit ratio of am asset's return series versus a target return series. If the target is not specified, 
    it is set to 0. The hit ratio is defined as the percentage of periods when the return on the asset 
    was greater than the target return.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param target_series: non cumulative target return series, if none it is set to 0, defaults to None
    :type target_series: pd.Series, optional
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional        
    :return: hit ratio
    :rtype: float
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    # Recursive Case
    if n_series > 1:
        return return_series.dropna(how='all'). \
            apply(lambda series: hit_ratio(series, target_series=target_series, dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)

    if target_series is None:
        target_series = pd.Series(
            np.zeros(return_series.shape[0]), index=return_series.index, name='Target')

    joined_data = return_series.join(target_series, how='left')
    return_series = joined_data.iloc[:, :1]
    target_series = joined_data.iloc[:, 1]
    # Check there are no nans in the target series
    if target_series.isna().sum().squeeze() > 0:
        print('Warning! Not enough Target returns data points.')
        return np.nan
    numerator = return_series[return_series.sub(
        target_series.values, axis=0) > 0].dropna().shape[0]
    denominator = return_series.shape[0]
    return numerator / denominator


def aggregate_monthly_returns(return_series: pd.Series):
    """ Aggregates monthly returns by month and year. Returns a dataframe with Year on the rows and Month on
    the columns.

    :param return_series: non cumulative monthly return series
    :type return_series: pd.Series
    :return: aggregate dataframe
    :rtype: pd.DataFrame
    """
    # Dealing with pd.DataFrame vs. pd.Series
    df = pd.DataFrame(return_series).copy(deep=True)
    df.dropna(inplace=True)
    series_name = df.columns[0]

    # Aggregate data by month and year
    date = df.index
    df['Year'] = date.strftime('%Y')
    # For ordering purposes, then it is changed to %b
    df['Month'] = date.strftime('%m')
    df_agg = df.pivot(index=['Year'], columns=['Month'], values=[
        series_name]).sort_index(ascending=False)
    # Dropping Fund Name Level from the MultiIndex
    df_agg.columns = df_agg.columns.droplevel(0)
    df_agg.columns = [calendar.month_abbr[int(
        col_name)] for col_name in df_agg.columns]  # %m to %b
    df_agg.rename_axis('Month', axis=1, inplace=True)

    # Add an Annual column
    df_agg['Annual'] = df_agg.apply(
        lambda row: (1 + row.dropna()).prod() - 1, axis=1)

    return df_agg


def tracking_error(return_series: pd.Series, benchmark_series: pd.Series, periods_per_year: int = 12,
                   dropna: bool = True):
    """Computes annualized tracking error of a series vs its benchmark.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param benchmark_series: non cumulative benchmark return series
    :type benchmark_series: pd.Series
    :param periods_per_year: inverse yearly frequency of observations, defaults to 12
    :type periods_per_year: int, optional
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional        
    :return: annualized tracking error
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    if n_series > 1:  # Recursive case to deal with multiple series
        return return_series.dropna(how='all'). \
            apply(lambda series: tracking_error(series, benchmark_series=benchmark_series,
                                                periods_per_year=periods_per_year, dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)

    joined_data = return_series.join(benchmark_series, how='left')
    return_series = joined_data.iloc[:, :1]
    benchmark_series = joined_data.iloc[:, 1]
    # Check there are no nans in the benchmark series
    if benchmark_series.isna().sum().squeeze() > 0:
        print('Warning! Not enough Benchmark returns data points.')
        return np.nan

    # Compute Excess Returns
    excess_returns = pd.DataFrame(return_series).sub(
        benchmark_series.values, axis=0)
    excess_returns = excess_returns.iloc[:, 0]

    # Compute tracking error
    return annualize_vol(excess_returns, periods_per_year=periods_per_year)


def total_return(return_series: pd.Series, dropna: bool = True) -> np.float64:
    """Total Return for period, not annualized.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional
    :return: Total Return
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    # Recursive Case
    if n_series > 1:
        return return_series.dropna(how='all').apply(lambda col: total_return(col, dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)

    return ((1 + return_series).prod() - 1).squeeze()


def information_ratio(return_series: pd.Series, benchmark_series: pd.Series, periods_per_year: int = 12,
                      dropna: bool = True):
    """Computes annualized information ratio of a return series against its benchmark.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param benchmark_series: non cumulative benchmark return series
    :type benchmark_series: pd.Series
    :param periods_per_year: inverse yearly frequency of observations, defaults to 12
    :type periods_per_year: int, optional
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional        
    :return: annualized information ratio
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    # Recursive Case
    if n_series > 1:
        return return_series.dropna(how='all'). \
            apply(lambda series: information_ratio(series, benchmark_series=benchmark_series,
                                                   periods_per_year=periods_per_year, dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan

    joined_data = return_series.dropna().join(benchmark_series, how='left')
    return_series = joined_data.iloc[:, :1]
    benchmark_series = joined_data.iloc[:, 1]

    # Check there are no nans in the bm series
    if benchmark_series.isna().sum().squeeze() > 0:
        print('Warning! Not enough Benchmark data points.')
        return np.nan

    diff = pd.DataFrame(return_series).sub(
        benchmark_series.values, axis=0).mean() * periods_per_year
    ir = diff / tracking_error(return_series, benchmark_series, periods_per_year)
    return ir.squeeze()


def downside_deviation(return_series: pd.Series, risk_free_rate: pd.Series = None, periods_per_year: int = 12,
                       dropna: bool = True) -> np.float64:
    """Computes annualized downside deviation of a return series.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param risk_free_rate: non cumulative risk-free rate return series, if None it is assumed to e 0, defaults to None
    :type risk_free_rate: pd.Series
    :param periods_per_year: inverse yearly frequency of observations, defaults to 12
    :type periods_per_year: int, optional
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional        
    :return: annualized downside deviation
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    if n_series > 1:  # Recursive case to deal with multiple series
        return return_series.dropna(how='all'). \
            apply(lambda series: downside_deviation(series, risk_free_rate=risk_free_rate,
                                                    periods_per_year=periods_per_year, dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)

    if risk_free_rate is not None:
        joined_data = return_series.join(risk_free_rate, how='left')
        return_series = joined_data.iloc[:, :1]
        risk_free_rate = joined_data.iloc[:, 1]
        # Check there are no nans in the rf series
        if risk_free_rate.isna().sum().squeeze() > 0:
            print('Warning! Not enough Risk Free returns data points.')
            return np.nan
    else:
        risk_free_rate = pd.Series(
            np.zeros(return_series.shape[0]), index=return_series.index, name='RF')

    # Compute Excess Returns
    excess_returns = pd.DataFrame(return_series).sub(
        risk_free_rate.values, axis=0)
    excess_returns = excess_returns.iloc[:, 0]
    n_periods = excess_returns.shape[0]

    # Remove positive excess returns
    excess_returns[excess_returns > 0] = 0
    sum_of_squares = np.power(excess_returns, 2).sum()
    dd = np.math.sqrt(sum_of_squares / (n_periods - 1)) * \
         np.math.sqrt(periods_per_year)
    return dd


def upside_deviation(return_series: pd.Series, risk_free_rate: pd.Series = None, periods_per_year: int = 12,
                     dropna: bool = True) -> np.float64:
    """Computes annualized upside deviation of a return series.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param risk_free_rate: non cumulative risk-free rate return series, if None it is assumed to e 0, defaults to None
    :type risk_free_rate: pd.Series
    :param periods_per_year: inverse yearly frequency of observations, defaults to 12
    :type periods_per_year: int, optional
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional        
    :return: annualized upside deviation
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    if n_series > 1:  # Recursive case to deal with multiple series
        return return_series.dropna(how='all'). \
            apply(lambda series: upside_deviation(series, risk_free_rate=risk_free_rate,
                                                  periods_per_year=periods_per_year, dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)

    if risk_free_rate is not None:
        joined_data = return_series.join(risk_free_rate, how='left')
        return_series = joined_data.iloc[:, :1]
        risk_free_rate = joined_data.iloc[:, 1]
        # Check there are no nans in the rf series
        if risk_free_rate.isna().sum().squeeze() > 0:
            print('Warning! Not enough Risk Free returns data points.')
            return np.nan
    else:
        risk_free_rate = pd.Series(
            np.zeros(return_series.shape[0]), index=return_series.index, name='RF')

    # Compute Excess Returns
    excess_returns = pd.DataFrame(return_series).sub(
        risk_free_rate.values, axis=0)
    excess_returns = excess_returns.iloc[:, 0]
    n_periods = excess_returns.shape[0]

    excess_returns[excess_returns < 0] = 0  # Remove negative excess returns
    sum_of_squares = np.power(excess_returns, 2).sum()
    ud = np.math.sqrt(sum_of_squares / (n_periods - 1)) * \
         np.math.sqrt(periods_per_year)
    return ud


def sortino_ratio(return_series: pd.Series, risk_free_rate: pd.Series = None,
                  periods_per_year: int = 12, dropna: bool = True) -> np.float64:
    """Computes annualized Sortino Ratio of a return series.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param risk_free_rate: non cumulative risk-free rate return series, if None it is set to 0, defaults to None
    :type risk_free_rate: pd.Series
    :param periods_per_year: inverse yearly frequency of observations, defaults to 12
    :type periods_per_year: int, optional
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional        
    :return: annualized Sortino Ratio
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    n_series = return_series.shape[1]

    if n_series > 1:  # Recursive case to deal with multiple series
        return return_series.dropna(how='all'). \
            apply(lambda series: sortino_ratio(series, risk_free_rate=risk_free_rate,
                                               periods_per_year=periods_per_year, dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan
    return_series.dropna(inplace=True)

    if risk_free_rate is not None:
        joined_data = return_series.join(risk_free_rate, how='left')
        return_series = joined_data.iloc[:, :1]
        risk_free_rate = joined_data.iloc[:, 1]
        # Check there are no nans in the rf series
        if risk_free_rate.isna().sum().squeeze() > 0:
            print('Warning! Not enough Risk Free returns data points.')
            return np.nan
    else:
        print('Warning! risk_free_rate was not provided, it will be set to 0. Results may not be reliable.')
        risk_free_rate = pd.Series(
            np.zeros(return_series.shape[0]), index=return_series.index, name='RF')

    # Like SR, we use arithmetic mean and not geometric mean, not to penalize on volatility also at the numerator
    r = return_series.mean() * periods_per_year
    r_f = risk_free_rate.mean() * periods_per_year
    diff = r - r_f
    dd = downside_deviation(return_series, risk_free_rate,
                            periods_per_year=periods_per_year)
    return (diff / dd).squeeze()


def up_down_beta(return_series: pd.Series, benchmark_series: pd.Series):
    """Returns the up and down beta of a return series against a benchmark series.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param benchmark_series: non cumulative benchmark return series
    :type benchmark_series: pd.Series
    :return: up-beta (beta conditional on positive benchmark returns) and down-beta (beta conditional on negative
     benchmark returns)
    :rtype: (np.float64, np.float64)
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series_name = return_series.columns if len(
        return_series.shape) > 1 else [return_series.name]
    benchmark_series_name = benchmark_series.columns if len(
        benchmark_series.shape) > 1 else [benchmark_series.name]
    # Data wrangling
    joined_data = pd.DataFrame(return_series).join(
        benchmark_series, how='inner').dropna()
    joined_data['up_down'] = joined_data.apply(
        lambda row: 'Up' if row[benchmark_series_name].values >= 0 else 'Down', axis=1)

    # Computations

    def compute_beta(df):
        model = sm.OLS(df[return_series_name], sm.add_constant(
            df[benchmark_series_name])).fit()
        return model.params.values[1]

    up_beta = compute_beta(
        joined_data.loc[joined_data['up_down'] == 'Up'].drop('up_down', axis=1))
    down_beta = compute_beta(
        joined_data.loc[joined_data['up_down'] == 'Down'].drop('up_down', axis=1))
    return up_beta, down_beta


def market_capture_ratio(return_series: pd.Series, benchmark_series: pd.Series, up: bool = True, dropna: bool = True):
    """Returns the up- (or down-) market capture ratio of a return series against a benchmark series.
    The up-market capture ratio is calculated by dividing the manager's returns by the returns of the 
    index during the up-market and multiplying that factor by 100.

    :param return_series: non cumulative return series
    :type return_series: pd.Series
    :param benchmark_series: non cumulative benchmark return series
    :type benchmark_series: pd.Series
    :param up: if True, the function returns the up-market capture ratio, else it returns the down-market capture ratio.
    :type up: bool    
    :param dropna: when False and nan values are supplied, the function will return itself a nan value.
    dropna = False is needed when 1) the input to the functions is a dataframe containing multiple return series
     (of different length) and 2) The computation refers to a specific time-span (e.g. L5Y Return), defaults to True
    :type dropna: bool, optional        
    :return: up-(down-) market capture ratio
    :rtype: np.float64
    """
    # Dealing with pd.Series vs. pd.DataFrame
    return_series = pd.DataFrame(return_series).copy(deep=True)
    benchmark_series = pd.DataFrame(benchmark_series)
    benchmark_series_name = benchmark_series.columns
    n_series = return_series.shape[1]

    if n_series > 1:  # Recursive case to deal with multiple series
        return return_series.dropna(how='all'). \
            apply(lambda col: market_capture_ratio(return_series=col, benchmark_series=benchmark_series, up=up,
                                                   dropna=dropna), axis=0)

    # Base Case
    if (not dropna) and (return_series.isna().sum().squeeze() > 0):
        return np.nan

    # Data wrangling
    joined_data = return_series.dropna().join(
        benchmark_series, how='left')

    if joined_data[benchmark_series_name].isna().sum().squeeze() > 0:
        print('Warning! Not enough Benchmark data points.')
        return np.nan

    joined_data['up_down'] = joined_data.apply(
        lambda row: 'Up' if row.loc[benchmark_series_name].values >= 0 else 'Down', axis=1)

    # Computations
    market_classification = 'Up' if up else 'Down'
    # Filter data by Up or down markets
    filtered_data = joined_data[joined_data['up_down']
                                == market_classification].drop('up_down', axis=1)
    total_returns = filtered_data.apply(lambda x: np.prod(1 + x) - 1, axis=0)

    return (total_returns.iloc[0] / total_returns.iloc[1]).squeeze() * 100


def portfolio_return_series(returns_df: pd.DataFrame, weights: np.array = None, rebalancing: bool = True):
    """Computes the return series of a portfolio made up by the assets in returns_df with weights set in weights.

    :param returns_df: dataframe of non cumulative return series, each column corresponds to a different asset
    :type returns_df: pd.DataFrame
    :param weights: portfolio weights if rebalanced otherwise portfolio initial weights, if None it is set to
    equally weighted, defaults to None
    :type weights: np.array, optional
    :param rebalancing: indicates whether the portfolio is rebalanced or not, defaults to True
    :type rebalancing: bool, optional
    :return: series of portfolio returns
    :rtype: pd.Series
    """
    # Data Wrangling
    returns_df = returns_df.copy(deep=True)
    returns_df.dropna(inplace=True)
    n_series = returns_df.shape[1]
    dates = returns_df.index

    if weights is None:
        weights = np.array([1/n_series]*n_series)

    if rebalancing:
        return_series = pd.Series(
            np.dot(returns_df, weights), index=dates, name='Portfolio')
        return return_series

    # Add a a row of ones at the beginning of cumulative_returns dataframe
    cumulative_returns = (1+returns_df).cumprod()
    cumulative_returns = pd.concat([pd.DataFrame(np.ones(n_series).reshape(
        1, n_series), columns=returns_df.columns), cumulative_returns])
    # Compute raw weights
    weights_raw = (weights * cumulative_returns).iloc[0:-1]
    # Normalize weights and readjust index to make it compatible to the return series
    weights = weights_raw.div(weights_raw.sum(axis=1), axis=0)
    weights.index = returns_df.index
    # Finally, compute returns
    return_series = pd.Series(
        (returns_df * weights).sum(axis=1), name='Portfolio')
    return return_series


if __name__ == '__main__':
    pass
