from ..visuals import *
import pandas as pd
from .analytics import aggregate_return_contributions, rolling_risk_attribution


def cumulative_return_contribution_plot(contribution_df: pd.DataFrame,
                                        title='Cumulative Returns - Contributions'):
    df = contribution_df.copy(deep=True).dropna(how='all').fillna(0)
    df_aggregate = df.copy(deep=True)
    dates = df.index

    # First, we need to aggregate attribution by each period
    for i, date in enumerate(dates):
        df_aggregate.loc[date, :] = aggregate_return_contributions(
            df.iloc[:i + 1], periods_per_year=12, annualized=False)

    # Now, we can plot
    fig = cumulative_returns_plot(
        df_aggregate, cumulative=True, start_from_zero=True, title=title)

    return fig


def rolling_risk_attribution_plot(contribution_df: pd.DataFrame,
                                  rolling_factor: int = 36,
                                  show_in_percentage: bool = False,
                                  title=None):
    contribution_df = contribution_df.copy(deep=True).dropna(how='all').fillna(0)
    risk_attribution_df = rolling_risk_attribution(annualize_vol, contribution_df, rolling_factor=rolling_factor,
                                                   show_in_percentage=show_in_percentage)
    if title is None:
        title_addition = ' (% of Total)' if show_in_percentage else ''
        title = f'Rolling {rolling_factor}M Volatility Attribution' + title_addition

    # Now, we can plot
    fig = time_series_plot(
        risk_attribution_df, title=title,
        yaxis_tick_format='.1%')

    return fig


if __name__ == '__main__':
    pass
