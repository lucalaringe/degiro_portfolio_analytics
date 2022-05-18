from .analytics import *
from .plotly_utils import *

import pandas as pd

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


def time_series_plot(df: pd.DataFrame, show_average: bool = False, yaxis_tick_format=',.2f',
                     xaxis_label=None, yaxis_label=None, title=None):
    """ Automatic time series plotting function in plotly.

    :param df: time series dataframe of related series to be plotted
    :type df: pd.DataFrame
    :param show_average: indicates whether to plot, as a dashed line of the same color of the corresponding series,
     the averages of the series, defaults to False
    :type show_average: bool, optional
    :param yaxis_tick_format: y axis tick format. Some examples are: {'.2f', '.0f', '.2%'}.
    :type yaxis_tick_format: str
    :param xaxis_label: x axis label to bbe shown in the chart, defaults to None
    :type xaxis_label: str
    :param yaxis_label: y axis label to bbe shown in the chart, defaults to None
    :type yaxis_label: str
    :param title: title of the chart, defaults to None
    :type title: str
    :return: plotly figure
    :rtype: plotly.graph_objs._figure.Figure
    """

    # Dealing with pd.DataFrame vs. pd.Series
    series_names = df.columns if len(df.shape) > 1 else [df.name]
    df = pd.DataFrame(df)
    df.dropna(how='all', inplace=True)
    colors = list(COLORS.values())[:len(series_names)]

    if show_average:
        for name in series_names:
            df[name + ' - Average'] = df[name].mean()

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    for name, color in zip(series_names, colors):
        fig.add_trace(
            go.Scatter(x=df.index, y=df[name], name=name, mode='lines',
                       line=dict(color=color, width=2)),
            secondary_y=False
        )
        if show_average:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[name + ' - Average'], name=name + ' - Average', mode='lines',
                           line=dict(dash='dash', color=color, width=2)),
                secondary_y=False
            )

    if len(series_names) == 1 and yaxis_label == '':
        yaxis_label = series_names[0]

    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(zeroline=False),
        yaxis=dict(zeroline=False),
        legend=dict(xanchor='center', x=0.5, orientation='h'),
        template=PLOTLY_TEMPLATE
    )

    fig.update_layout(
        yaxis=dict(tickformat=yaxis_tick_format)
    )

    if xaxis_label is not None:
        fig.update_layout(xaxis=dict(title='{}'.format(xaxis_label)))
    if title is not None:
        fig.update_layout(title=title)
    if yaxis_label is not None:
        fig.update_layout(yaxis=dict(title='{}'.format(yaxis_label)))

    return fig


def area_plot(df: pd.DataFrame, x: str, y: str, color: str, line_group: str = None, yaxis_tick_format=',.2f',
              xaxis_label=None, yaxis_label=None, title=None):
    """ Automatic area plotting function in plotly. DataFrame must be in long-form with no index.

    :param line_group: line_group variable name
    :param color: color variable name
    :param x: x variable name
    :param y: y variable name
    :param df: time series dataframe of related series to be plotted
    :type df: pd.DataFrame
    :param yaxis_tick_format: y axis tick format. Some examples are: {'.2f', '.0f', '.2%'}.
    :type yaxis_tick_format: str
    :param xaxis_label: x axis label to bbe shown in the chart, defaults to None
    :type xaxis_label: str
    :param yaxis_label: y axis label to bbe shown in the chart, defaults to None
    :type yaxis_label: str
    :param title: title of the chart, defaults to None
    :type title: str
    :return: plotly figure
    :rtype: plotly.graph_objs._figure.Figure
    """
    # Dealing with pd.DataFrame vs. pd.Series
    df = pd.DataFrame(df)
    df.dropna(how='all', inplace=True)
    colors = list(COLORS.values())

    fig = px.area(df, x=x, y=y, color=color, line_group=line_group, color_discrete_sequence=colors)

    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(title=None, zeroline=False),
        yaxis=dict(title=None, zeroline=False),
        legend=dict(title=None, xanchor='center', x=0.5, orientation='h'),
        template=PLOTLY_TEMPLATE
    )

    fig.update_layout(
        yaxis=dict(tickformat=yaxis_tick_format)
    )

    if xaxis_label is not None:
        fig.update_layout(xaxis=dict(title='{}'.format(xaxis_label)))
    if title is not None:
        fig.update_layout(title=title)
    if yaxis_label is not None:
        fig.update_layout(yaxis=dict(title='{}'.format(yaxis_label)))

    return fig


def pie_chart(series: pd.Series, percentage: bool = True, title: str = None, show_legend: bool = True,
              sort: bool = False, hovertemplate_tick_format: str = ',.1f', percentage_precision: int = 1):
    """Plots a Pie chart of a pandas series.

    :param percentage_precision: if percentage is picked, number of decimals to display
    :param hovertemplate_tick_format: format to display numbers when hovering
    :param sort: indicates whether to sort the elements in descending order, defaults to True
    :param series: pandas series
    :type series: pd.Series
    :param percentage: indicates whether to show numbers as a % of total, defaults to True
    :type percentage: bool, optional
    :param title: chart title, defaults to None
    :type title: str, optional
    :param show_legend: indicates whether to display the legend, defaults to True
    :type show_legend: bool, optional
    :return: pie chart
    :rtype: plotly.graph_objs._figure.Figure
    """
    index_name = series.index.name
    variable_name = series.name
    if sort:
        series.sort_values(ascending=False, inplace=True)
    data = pd.DataFrame(series).reset_index(drop=False)
    fig = px.pie(data,
                 names=index_name,
                 color=index_name,
                 values=variable_name,
                 hole=0.5,
                 opacity=0.8,
                 color_discrete_sequence=list(COLORS.values())
                 )
    hovertemplate_idx = [str(idx)+': ' for idx in data[index_name]]
    hovertemplate_values = data[variable_name].apply(lambda x: f'{x:{hovertemplate_tick_format}}')
    fig.update_traces(hovertemplate=[i+j for i, j in zip(hovertemplate_idx, hovertemplate_values)])

    if percentage:
        fig.update_traces(texttemplate=(data[variable_name]/data[variable_name].sum()).
                          apply(lambda x: f'{x:.{percentage_precision}%}'))

    fig.update_layout(showlegend=False, uniformtext=dict(minsize=14, mode='hide'),
                      margin=dict(l=0, r=0, t=30, b=0),
                      template=PLOTLY_TEMPLATE)
    fig.update_traces(textposition='inside')

    if title is not None:
        fig.update_layout(title=dict(text=title, x=0.5))
    if not percentage:
        fig.update_traces(textinfo='value')
    if show_legend:
        fig.update_layout(showlegend=True,
                          legend=dict(orientation="h", xanchor="center", x=0.5)
                          )
    return fig


def bar_grouped_plot(df: pd.DataFrame, x: str, y: str, color: str, barmode: str = 'group',
                     show_legend_at_bottom: bool = False, yaxis_tick_format: str = ',.2f',
                     xaxis_label: str = None, yaxis_label: str = None, legend_title: str = None,
                     title: str = None):
    """Grouped Bar Plot of a pandas dataframe.

    :param legend_title: legend title
    :param df: pandas dataframe to plot (in long form)
    :type df: pd.DataFrame
    :param x: x-axis column name
    :type x: str
    :param y: y-axis column name
    :type y: str
    :param color: category column name
    :type color: str
    :param barmode: barmode type. Must be in {'group', 'overlay', 'relative'}, defaults to 'group'
    :type barmode: str, optional
    :param show_legend_at_bottom: indicates whether to show the legend on the bottom of the chart, defaults to False
    :type show_legend_at_bottom: bool, optional
    :param yaxis_tick_format: y axis tick format. Some examples are: {'.2f', '.0f', '.2%'}.
    :type yaxis_tick_format: str
    :param xaxis_label: x axis label to be shown in the chart, defaults to None
    :type xaxis_label: str
    :param yaxis_label: y axis label to be shown in the chart, defaults to None
    :type yaxis_label: str
    :param title: title of the legend, defaults to None
    :type title: str
    :param title: title of the chart, defaults to None
    :type title: str
    :return: bar plot of the series
    :rtype: plotly.graph_objs._figure.Figure
    """
    # Visualization
    n_variables = df[color].unique().shape[0]
    color_discrete_map = dict(zip(df[color].unique().tolist(), list(COLORS.values())[:n_variables]))

    fig = px.bar(df, x=x, y=y, color=color, barmode=barmode, color_discrete_map=color_discrete_map)

    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(type='category', title=None),
        yaxis=dict(tickformat=yaxis_tick_format, title=None),
        showlegend=True,
        legend=dict(title=None),
        template=PLOTLY_TEMPLATE
    )

    if title is not None:
        fig.update_layout(title=f'{title}')
    if xaxis_label is not None:
        fig.update_layout(xaxis=dict(title=f'{xaxis_label}'))
    if yaxis_label is not None:
        fig.update_layout(yaxis=dict(title=f'{yaxis_label}'))
    if legend_title is not None:
        fig.update_layout(legend=dict(title=f'{color}'))
    if show_legend_at_bottom:
        fig.update_layout(legend=dict(xanchor='center', x=0.5, orientation='h'))

    return fig


def cumulative_returns_plot(return_series: pd.DataFrame, cumulative: bool = False, start_from_zero: bool = True,
                            title: str = None):
    """Plots cumulative returns of one or many returns series.

    :param return_series: one/multiple (non-cumulative) return series
    :type return_series: pd.DataFrame
    :param cumulative: indicated whether the return series supplied is/are cumulative, defaults to False
    :type cumulative: bool, optional
    :param start_from_zero: Indicates whether all cumulative series should start from zero 1 month before the
    first date, defaults to True
    :type start_from_zero: bool, optional
    :param title: title of the chart, defaults to None
    :type title: str, optional
    :return: plot
    :rtype: plotly.graph_objs._figure.Figure
    """
    # Dealing with pd.DataFrame vs. pd.Series
    return_series = pd.DataFrame(return_series).copy(
        deep=True).dropna(how='all')
    n_series = return_series.shape[1]
    if title is None:
        if n_series <= 1:
            title = f'{return_series.columns[0]} - ' + 'Cumulative Returns'
        else:
            title = 'Cumulative Returns'

    if start_from_zero:
        # Add 0 return at the beginning of the series
        full_names = []
        na_names = []
        for col_name in return_series.columns:
            if return_series[col_name].dropna().shape[0] == return_series.shape[0]:
                full_names.append(col_name)
            else:
                na_names.append(col_name)
        # Append 0 at the beginning for series with no nan
        if len(full_names) > 0:
            return_series.loc[return_series.index[0] +
                              pd.DateOffset(months=-1), full_names] = 0
            return_series.sort_index(inplace=True)
        # Append 0 at the beginning for series with nans
        if len(na_names) > 0:
            for name in na_names:
                idx = np.where(return_series.index ==
                               return_series[na_names].first_valid_index())[0][0]
                return_series.loc[return_series.index[idx - 1], name] = 0

    # Compute cumulative returns
    if not cumulative:
        cumulative_returns = (return_series + 1).cumprod() - 1
    else:
        cumulative_returns = return_series

    fig = time_series_plot(
        cumulative_returns, yaxis_tick_format='.1%', title=title)

    return fig


def drawdown_comparison_plot(return_df: pd.DataFrame, title: str = 'Drawdowns'):
    """Given a return dataframe, plots the drawdowns of the different series side by side, allowing for
    comparison of drawdowns in stress periods.

    :param return_df: non-cumulative return series
    :type return_df: pd.DataFrame
    :param title: title of the chart, defaults to 'Drawdowns'
    :type title: str, optional
    :return: drawdown plot
    :rtype: plotly.graph_objs._figure.Figure
    """

    cumulative = (1 + return_df).cumprod()
    peaks = cumulative.cummax()
    drawdowns = ((cumulative / peaks) - 1)

    # Figure
    fig = time_series_plot(drawdowns, title=title)
    fig.update_layout(
        hovermode='x unified',
        yaxis=dict(title=None,
                   tickformat=".1%", zeroline=False))

    return fig


def rolling_volatility_plot(return_series: pd.DataFrame, rolling_factor: int = 36, show_full_sample: bool = False,
                            title: str = None):  # Assumes monthly data
    """Plots annualized rolling volatility of one or many returns series.

    :param return_series: one/multiple non-cumulative return series
    :type return_series: pd.DataFrame
    :param rolling_factor: rolling window in periods (months), defaults to 36
    :type rolling_factor: int, optional
    :param show_full_sample: indicates whether to plot, as a dashed line of the same color of the corresponding series,
    the full-sample measure, defaults to False
    :type show_full_sample: bool, optional
    :param title: title of the chart, defaults to None
    :type title: str, optional
    :return: plot
    :rtype: plotly.graph_objs._figure.Figure
    """
    # Dealing with pd.DataFrame vs. pd.Series
    return_series = pd.DataFrame(return_series).copy(deep=True)
    series_names = list(return_series.columns)

    # Compute rolling volatility
    df = return_series.rolling(rolling_factor).apply(
        annualize_vol, raw=False)
    df.dropna(how='all', inplace=True)

    if title is None:
        title = f'Rolling {rolling_factor}M Volatility'

    fig = time_series_plot(
        df, show_average=False, yaxis_tick_format='.1%', title=title)

    if show_full_sample:
        # Annualize Volatility over full-sample
        full_sample_vol = return_series.apply(
            annualize_vol, axis=0, raw=False)
        for name in series_names:
            df[name + ' - Full Sample'] = full_sample_vol[name]
        colors = list(COLORS.values())

        for i, name in enumerate(series_names):
            fig.add_trace(
                go.Scatter(x=df.index, y=df[name + ' - Full Sample'], name=name + ' - Full Sample', mode='lines',
                           line=dict(dash='dash', color=colors[i], width=2)),
                secondary_y=False
            )
    return fig


def returns_distribution_plot(return_series: pd.Series, marginal: str = 'violin', nbins: int = 30, title: str = None):
    """Plots returns histogram (+ kernel density estimation).

    :param return_series: non-cumulative return series
    :type return_series: pd.Series
    :param marginal: Indicates the type of marginal plot, can be in ('box', 'violin', 'rug', None), defaults to 'violin'
    :type marginal: str, optional
    :param nbins: Number of bins to use in the histogram, defaults to 30
    :type nbins: int, optional
    :param title: title of the chart, defaults to None
    :type title: str, optional
    :return: plot
    :rtype: plotly.graph_objs._figure.Figure
    """
    # Dealing with pd.DataFrame vs. pd.Series
    df = pd.DataFrame(return_series).copy(deep=True)
    df = df.rename_axis('Series', axis=1)

    if title is None:
        title = f'Return Distribution'

    fig = px.histogram(df, nbins=nbins, marginal=marginal,
                       color_discrete_sequence=list(COLORS.values()), histnorm='percent')

    fig.update_layout(
        hovermode='x unified',
        title=title,
        xaxis=dict(title=None, zeroline=False, tickformat=".1%"),
        yaxis=dict(title='% of Observations',
                   zeroline=False, tickformat=".1f%"),
        legend=dict(title=None, xanchor='center', x=0.5, orientation='h'),
        template=PLOTLY_TEMPLATE
    )

    return fig


def horizontal_bar_plot(series: pd.Series, xaxis_tick_format: str = ',.2f', xaxis_label=None, yaxis_label=None,
                        title=None):
    """ Creates a horizontal bar plot for a pandas series.

    :param series: series to plot
    :param xaxis_tick_format: specifies the tick format for the x-axis
    :param xaxis_label: x axis label, defaults to None
    :param yaxis_label: y axis label, defaults to None
    :param title: chart's title
    :return: horizontal bar chart
    """

    series_names = series.index
    colors = list(COLORS.values())[:len(series_names)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=series.index,
        x=series.values,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color=colors, width=3)
        )
    ))

    fig.update_layout(template=PLOTLY_TEMPLATE,
                      showlegend=False,
                      yaxis=go.layout.YAxis(
                          tickangle=-40),
                      xaxis=dict(tickformat=xaxis_tick_format))

    if xaxis_label is not None:
        fig.update_layout(xaxis=dict(title=xaxis_label))
    if title is not None:
        fig.update_layout(title=title)
    if yaxis_label is not None:
        fig.update_layout(yaxis=dict(title=yaxis_label))

    return fig


def correlation_plot(return_series: pd.DataFrame, title: str = None,
                     mask_upper_triangle: bool = False):
    """Plots correlation of multiple return series in a heatmap.

    :param return_series: one or multiple non cumulative return series
    :type return_series: pd.DataFrame
    :param title: title of the chart: can be overridden, defaults to None
    :type title: str, optional
    :param mask_upper_triangle: show upper triangle as blank (avoiding duplicate), default is False
    :type mask_upper_triangle: bool, optional
    :return: figure
    :rtype: plotly.graph_objs._figure.Figure
    """
    # Data Manipulation
    df = pd.DataFrame(return_series).copy(deep=True).dropna(how='all')

    df_corr = df.corr()
    if mask_upper_triangle:
        mask = np.zeros_like(df_corr, dtype=np.bool)
        mask[np.triu_indices_from(mask, 1)] = True
        df_corr = df_corr.mask(mask)

    # Plotting
    if title is None:
        title = f'Correlations'

    z = df_corr.to_numpy()
    z_text = np.around(z, decimals=2)
    if mask_upper_triangle:
        z_text = [[z_text[i][j] if i > j else '' for j in range(
            z_text.shape[0])] for i in range(z_text.shape[1])]

    fig = ff.create_annotated_heatmap(
        z=z,
        x=df_corr.columns.tolist(),
        y=df_corr.index.tolist(),
        annotation_text=z_text,
        zmax=1, zmin=-1,
        showscale=True,
        hoverongaps=True,
        colorscale=GRWHRE,
        reversescale=True,
        font_colors=[COLORS['black'], COLORS['black']]
    )
    fig.update_yaxes(autorange="reversed")

    fig.update_layout(title=dict(text=title, xanchor='center', x=0.5),
                      xaxis=dict(showline=False),
                      yaxis=dict(showline=True, showspikes=True,
                                 zeroline=True),
                      hoverlabel=dict(font=dict(family=FONT), align='left'))

    fig.update(data=[
               {'hovertemplate': 'x: %{x}<br>y: %{y}<br>Correlation: %{z:.2f}<extra></extra>'}])
    fig.layout.coloraxis.showscale = True
    fig.update_layout(plot_bgcolor=opacitize('grey', 0.4))
    fig.update_coloraxes(colorbar=dict(outlinewidth=0, x=.765))

    return fig


def horizontal_waterfall_plot(series: pd.Series, xaxis_tick_format: str = ',.1%',
                              xaxis_label=None, yaxis_label=None, title=None):
    """
    Horizontal waterfall plot od a Pandas Series.
    Useful to show the positive and negative factors affecting some variable.

    :param series: series to plot
    :param xaxis_tick_format: specifies the tick format for the x-axis
    :param xaxis_label: x axis label, defaults to None
    :param yaxis_label: y axis label, defaults to None
    :param title: chart's title
    :return: horizontal waterfall plot
    """
    fig = go.Figure()

    fig.add_trace(go.Waterfall(
        orientation='h', measure=['relative', 'relative', 'relative', 'relative', 'relative', 'relative',
                                  'relative', 'relative', 'relative', 'relative', 'relative', 'relative'],
        y=series.index, x=series.values,
        connector={"mode": "between", "line": {"width": 1, "color": COLORS['black'], "dash": "solid"}},
        decreasing={"marker": {"color": COLORS['red'], "line": {"color": COLORS['red'], "width": 1}}},
        increasing={"marker": {"color": COLORS['green'], 'line': {'color':  COLORS['green'], 'width': 1}}}
    ))

    fig.update_layout(template=PLOTLY_TEMPLATE,
                      showlegend=False,
                      yaxis=go.layout.YAxis(
                          tickangle=-40),
                      xaxis=dict(tickformat=xaxis_tick_format)
                      )
    if xaxis_label is not None:
        fig.update_layout(xaxis=dict(title=xaxis_label))
    if title is not None:
        fig.update_layout(title=title)
    if yaxis_label is not None:
        fig.update_layout(yaxis=dict(title=yaxis_label))

    return fig


if __name__ == '__main__':
    pass
