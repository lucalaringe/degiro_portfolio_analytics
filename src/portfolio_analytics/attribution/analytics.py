from ..analytics import *
import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime


def risk_attribution(measure, components_returns: pd.DataFrame, weights: np.array = None, homogeneity_degree: int = 1,
                     show_in_percentage: bool = False, measure_name: str = 'Measure', *args) -> pd.Series:
    """Computes portfolio's risk attribution using Euler's Decomposition. This method can only be applied for positive
     homogeneous risk measures (e.g. VaR, Expected Shortfall, Volatility). The other inputs are a dataframe of
     non-cumulative return-series of the assets in the portfolio and the weights they are assigned.
    If weights is set to None, then it is assumed the input returns are returns' attribution
    (if you sum them up you get the portfolio's return, which means weights is a vector of ones).

    :param measure: risk measure function that takes as an input a pd.Series and returns a float. Additional args can be
    supplied to measure through *args
    :type measure: function
    :param components_returns: returns of the components of the portfolio
    :type components_returns: pd.DataFrame
    :param weights: weights of the components of the portfolio, if None it assumes returns are provided in form of
    attributions, defaults to None
    :type weights: np.array, optional
    :param homogeneity_degree: degree of homogeneity of the risk measure chosen, for instance VaR and vol are
     homogeneous of degree one, defaults to 1
    :type homogeneity_degree: int, optional
    :param show_in_percentage: Indicates whether to output the risk attribution in % of the total vs.
    in absolute value, defaults to False
    :type show_in_percentage: bool, optional
    :param measure_name: name of the output series, defaults to 'Measure'
    :type measure_name: str, optional
    :return: risk measure attribution series. Each entry corresponds to the amount of risk (as measured by measure)
    brought by that portfolio's component
    :rtype: pd.Series
    """
    df = components_returns.copy(deep=True).dropna(how='all')
    n_series = df.shape[1]
    series_names = df.columns

    if weights is None:
        weights = np.ones(n_series)  # returns_df contains return attributions

    # Define auxiliary function to compute partial derivatives
    def func(w):
        # Reconstruct Portfolio Returns
        portfolio_returns = portfolio_return_series(
            df, w, rebalancing=True)
        # Compute Portfolio Risk Measure
        return measure(portfolio_returns, *args)

    # Compute the gradient and the marginal contributions
    gradient = approx_fprime(xk=weights, f=func, epsilon=0.01)
    marginal_contributions = (1 / homogeneity_degree) * (gradient * weights)

    if show_in_percentage:
        marginal_contributions = marginal_contributions / marginal_contributions.sum()

    # Package output in a pd.Series
    out = pd.Series(
        dict(zip(series_names, marginal_contributions)), name=measure_name)

    return out


def rolling_risk_attribution(measure, components_returns: pd.DataFrame, weights: np.array = None,
                             rolling_factor: int = 36, homogeneity_degree: int = 1, show_in_percentage: bool = False,
                             measure_name: str = 'Measure', *args) -> pd.Series:
    """Computes portfolio's risk attribution using Euler's Decomposition on a rolling basis.
    This method can only be applied for positive homogeneous risk measures (e.g. VaR, Expected Shortfall, Volatility).
    The other inputs are a dataframe of non-cumulative return-series of the assets in the portfolio and the weights they
     are assigned.If weights is set to None, then it is assumed the input returns are returns' attribution
     (if you sum them up you get the portfolio's return, which means weights is a vector of ones).

    :param rolling_factor: rolling risk measure factor.
    :param measure: risk measure function that takes as an input a pd.Series and returns a float. Additional args can be
     supplied to measure through *args
    :type measure: function
    :param components_returns: returns of the components of the portfolio
    :type components_returns: pd.DataFrame
    :param weights: weights of the components of the portfolio, if None it assumes returns are provided in form of
     attributions, defaults to None
    :type weights: np.array, optional
    :param weights: length of the rolling window, defaults to 36
    :type weights: int, optional
    :param homogeneity_degree: degree of homogeneity of the risk measure chosen, for instance VaR and vol are
    homogeneous of degree one, defaults to 1
    :type homogeneity_degree: int, optional
    :param show_in_percentage: Indicates whether to output the risk attribution in % of the total vs. in absolute value,
     defaults to False
    :type show_in_percentage: bool, optional
    :param measure_name: name of the output series, defaults to 'Measure'
    :type measure_name: str, optional
    :return: risk measure attribution dataframe. Each column corresponds to the amount of risk (as measured by measure)
     brought by that portfolio's component
    :rtype: pd.Series
    """
    df = components_returns.copy(deep=True).dropna(how='all')
    n_samples = df.shape[0]
    dates = df.index
    series_names = df.columns
    out = pd.DataFrame(columns=series_names, index=dates[rolling_factor - 1:])

    if n_samples < rolling_factor:
        print(
            f'Warning! Not enough data to carry out the computation required (minimum: {rolling_factor}, '
            f'current: {n_samples}). Consider lowering the rolling factor or increasing the number of samples.')
        return None

    for i in range(0, n_samples - rolling_factor + 1):
        df_subset = df.iloc[i: i + rolling_factor, :]
        out.iloc[i, :] = risk_attribution(
            measure, df_subset, weights, homogeneity_degree, show_in_percentage, measure_name, *args)

    return out


def factor_volatility_attribution(dependent_variable, factors) -> pd.Series:
    """ Calculates the %attribution total portfolio volatility attributable to factor exposure (CRisk).
    Methodology is taken from Grinold and Kahn (1999). CRisk helps breakdown the portfolio risk by its systematic and
    idiosyncratic components.

    :param dependent_variable: dependent variable
    :param factors: exogenous factors
    :return: series of factor volatility attribution
    """
    factor_return_attribution_df = factor_return_attribution(dependent_variable, factors)
    portfolio_return_df = factor_return_attribution_df.sum(axis=1)

    factor_risk_attribution_series = factor_return_attribution_df.apply(
        lambda col: (col.std()*col.corr(portfolio_return_df)/portfolio_return_df.std()), axis=0)

    return factor_risk_attribution_series


def aggregate_return_contributions(contribution_df: pd.DataFrame, periods_per_year: int = 12, annualized: bool = False):
    """Given a set of single periods returns attributions, aggregates them into the full-period attribution. Note that 
    in order to maintain their additivity, return attributions cannot be aggregated the same way simple returns are.
    On one hand, note that in order to aggregate through time one series of returns you generally need only need that
    series. On the other hand, note that in order to aggregate a particular asset's return attribution through time,
    you need both the asset's return attribution and the full portfolio return over time.

    :param contribution_df: dataframe of return attributions to aggregate
    :type contribution_df: pd.DataFrame
    :param periods_per_year: inverse yearly frequency of observations, defaults to 12
    :type periods_per_year: int, optional
    :param annualized: indicates whether to annualize the full-period return attributions, defaults to False
    :type annualized: bool, optional
    :return: series of full-period return attributions
    :rtype: pd.Series
    """
    df = contribution_df.copy(deep=True).dropna(how='all')
    n_series = df.shape[1]
    n_periods = df.shape[0]

    # Attribution returns are aggregated by summing up components
    weights = np.ones(n_series)  # returns_df contains return attributions
    # Reconstruct Portfolio Returns
    portfolio_returns = portfolio_return_series(
        df, weights, rebalancing=True)
    # Generates weights to aggregate attributions
    portfolio_returns_gross = 1 + portfolio_returns
    factors = np.array(
        [1.] + portfolio_returns_gross[:-1].values.tolist()).cumprod()
    df_adjusted = df.apply(lambda col: col*factors, axis=0)

    result = df_adjusted.sum(axis=0)

    if annualized:
        total_return_portfolio = total_return(portfolio_returns)
        total_return_portfolio_ann = np.power(
            1+total_return_portfolio, periods_per_year/n_periods)-1

        result = (result/total_return_portfolio)*total_return_portfolio_ann

    return result


# Assumes monthly data
def attribution_stats(contribution_df: pd.DataFrame, show_total: bool = True, styled: bool = False):
    """Given a set of single periods returns attributions, aggregates them into MTD, QTD, YTD return attributions and
    L12M and ITD risk (vol) and return annualized attributions.

    :param contribution_df: dataframe of return attributions to aggregate
    :type contribution_df: pd.DataFrame
    :param show_total: indicates whether to show a total row in the final dataframe, defaults to True
    :type show_total: bool, optional
    :param styled: indicates whether to return a styled, more readable dataframe, defaults to False
    :type styled: bool, optional
    :return: aggregated return attributions dataframe
    :rtype: pd.DataFrame
    """
    periods_per_year = 12  # This function assumes monthly data
    contribution_df = contribution_df.copy(deep=True).dropna(how='all').fillna(0)

    # Current year and quarter
    year = contribution_df.index[-1].year
    quarter = contribution_df.index[-1].quarter

    def func_0(df): return df.iloc[-1]
    func_0.__name__ = 'MTD Returns'

    def func_1(df): return (
        (aggregate_return_contributions(df[(df.index.quarter == quarter) &
                                                        (df.index.year == year)])))
    func_1.__name__ = 'QTD Returns'

    def func_2(df): return (
        (aggregate_return_contributions(df[df.index.year == year])))
    func_2.__name__ = 'YTD Returns'

    def func_3(df):
        # Not enough data
        if df.dropna().shape[0] < 12:
            return pd.Series([np.nan], index=df.columns)
        return aggregate_return_contributions(df.iloc[-12:])
    func_3.__name__ = 'L12M Returns'

    def func_4(df):
        # Not enough data
        if contribution_df.dropna().shape[0] < 12:
            return pd.Series([np.nan], index=df.columns)
        return risk_attribution(
            annualize_vol, df.iloc[-12:])
    func_4.__name__ = 'L12M Risk'

    def func_5(df): return aggregate_return_contributions(
        df, periods_per_year=periods_per_year, annualized=True)
    func_5.__name__ = 'ITD Returns (Ann.)'

    def func_6(df): return risk_attribution(
        annualize_vol, df)
    func_6.__name__ = 'ITD Risk (Ann.)'

    func_list = [func_0, func_1, func_2, func_3, func_4, func_5, func_6]
    func_name_list = [func.__name__ for func in func_list]

    series = [pd.Series(func(contribution_df), name=func_name)
              for func, func_name in zip(func_list, func_name_list)]
    result = pd.concat(series, axis=1)

    if show_total:
        result.loc['Total', :] = result.sum(axis=0)

    # Styling
    if styled:
        col_styles = {}
        for col in result.columns:
            col_styles[col] = '{:,.2%}'
        result = result.style.format(col_styles, na_rep="-")

    return result


def factor_return_attribution(return_series: pd.Series, factors: pd.DataFrame, risk_free_rate: pd.Series = None):
    """
    Computes and outputs the Factor Return Attribution DataFrame.

    :param return_series: series of returns from an investment to be analyzed
    :type return_series: pd.Series
    :param factors: DataFrame of factor returns
    :type factors: pd.DataFrame
    :param risk_free_rate: time-series of risk-free rate asset returns (e.g. libor). It is subtracted to return_series
    if not None, defaults to None
    :type risk_free_rate: pd.Series, optional
    :return: DataFrame containing factor return attribution
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

    # Compute Factor Returns Attribution
    regression_df = regression(y, factors)
    coef = regression_df['coef'].values[1:]  # Exclude Alpha from the coefficients
    factor_attribution = factors.multiply(coef, axis=1)
    factor_attribution['Residual'] = y.subtract(factor_attribution.sum(axis=1))

    return factor_attribution


if __name__ == '__main__':
    pass
