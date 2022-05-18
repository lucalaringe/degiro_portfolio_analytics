from datetime import datetime
import pandas as pd

import constants
from utils import numeric_to_bps_str_format, numeric_to_percentage_str_format, \
    numeric_to_currency_str_format, numeric_to_int_str_format, numeric_to_float_str_format
import db_utils as db
import portfolio_analytics as pa
from portfolio_analytics import plotly_utils as plt

import dash
from dash_daq import BooleanSwitch
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# Constants
PRECISION_CUR = 0
PRECISION_DECIMAL = 2
PRECISION_PCT = 2
GROUPBY_FIELD_TO_LABEL = dict(zip(['product_id', 'name', 'asset_class'],
                                  ['DEGIRO Product Id', 'Product Name', 'Asset Class']))
FREQUENCY_TO_LABEL = {'1Y': 'Yearly', '1M': 'Monthly', '1D': 'Daily'}
TEXT_STYLE = {'font-family': 'Segoe UI', 'font-size': '95%'}
TABLE_HEADER_COLOR = plt.COLORS['blue']

# Load initial data
portfolio_nav = db.load_portfolio_nav()
portfolio_cash_flows = db.load_portfolio_cash_flows()
portfolio_returns = db.load_portfolio_returns(frequency='1M')
product_returns = db.load_product_return_time_series(frequency='1M')
market_data_details = db.load_market_data_details()
factor_data_details = db.load_factor_data_details()
market_series_returns = db.load_market_return_series(frequency='1M', currency='Local')
factor_series_returns = db.load_factor_return_series(frequency='1M')

portfolio_asset_nav = db.load_portfolio_asset_nav()
start_date = portfolio_asset_nav.index.min().date().strftime(constants.DATE_FORMAT)
end_date = portfolio_asset_nav.index.max().date().strftime(constants.DATE_FORMAT)
start_date_monthly = portfolio_returns.index.min().date().strftime(constants.DATE_FORMAT)
end_date_monthly = portfolio_returns.index.max().date().strftime(constants.DATE_FORMAT)

custom_sort_products = portfolio_asset_nav.loc[end_date].sort_values(ascending=False).index.values.tolist()


def pnl_contribution_table():
    portfolio_asset_pnl_df = db.load_portfolio_asset_pnl(frequency='1M').loc[start_date_monthly:end_date_monthly]
    portfolio_asset_returns_df = db.load_portfolio_asset_returns(frequency='1M') \
                                     .loc[start_date_monthly:end_date_monthly]
    current_month = datetime.strptime(end_date_monthly, constants.DATE_FORMAT).month
    qtd_n_monthly_obs = current_month % 3 if current_month % 3 != 0 else 3

    n_monthly_obs_mapping = {'MTD': 1, 'QTD': qtd_n_monthly_obs, 'YTD': current_month, 'L12M': 12, 'L3Y (Ann.)': 36,
                             'ITD (Ann.)': 9999}

    # Aggregate Time Weighted Returns
    def _time_weighted_returns(series: pd.Series, n_monthly_obs: int):
        # Annualize vs. not annualize
        annualized = True if n_monthly_obs > 12 else False
        dropna = True if n_monthly_obs >= 9999 else False
        r = series.iloc[-n_monthly_obs:]

        if annualized:
            return pa.annualize_returns(r, dropna=dropna)
        return pa.total_return(r, dropna=dropna)

    portfolio_asset_agg_time_weighted_returns_df = portfolio_asset_returns_df.apply(
        lambda series: pd.Series({key: _time_weighted_returns(series, value) for key, value in
                                  n_monthly_obs_mapping.items()})).T
    portfolio_agg_time_weighted_returns_df = pd.DataFrame(portfolio_returns).apply(
        lambda series: pd.Series({key: _time_weighted_returns(series, value) for key, value in
                                  n_monthly_obs_mapping.items()})).T

    # Aggregate P&L
    def _aggregate_pnl(series: pd.Series, n_monthly_obs: int):
        r = series.iloc[-n_monthly_obs:]
        return r.sum()

    portfolio_asset_agg_pnl_df = portfolio_asset_pnl_df.apply(
        lambda series: pd.Series({key: _aggregate_pnl(series, value) for key, value in
                                  n_monthly_obs_mapping.items()})).T

    # Generate Return Contributions
    portfolio_asset_agg_pnl_pct_df = portfolio_asset_agg_pnl_df.divide(portfolio_asset_agg_pnl_df.sum(axis=0))
    portfolio_asset_agg_contribution_df = portfolio_asset_agg_pnl_pct_df.multiply(
        portfolio_agg_time_weighted_returns_df.values[0], axis=1)
    portfolio_asset_agg_contribution_df = portfolio_asset_agg_contribution_df

    # Add Totals
    portfolio_asset_agg_pnl_df = portfolio_asset_agg_pnl_df
    portfolio_asset_agg_pnl_df.loc['Total', :] = portfolio_asset_agg_pnl_df.sum(axis=0)
    portfolio_asset_agg_time_weighted_returns_df.loc['Total', :] = portfolio_agg_time_weighted_returns_df.values[0]
    portfolio_asset_agg_contribution_df.loc['Total', :] = portfolio_asset_agg_contribution_df.sum(axis=0)

    # Format
    portfolio_asset_agg_pnl_df = portfolio_asset_agg_pnl_df.applymap(numeric_to_currency_str_format)
    portfolio_asset_agg_time_weighted_returns_df = portfolio_asset_agg_time_weighted_returns_df.applymap(
        numeric_to_percentage_str_format)
    portfolio_asset_agg_contribution_df = portfolio_asset_agg_contribution_df.applymap(
        numeric_to_bps_str_format)

    # MultiIndex
    portfolio_asset_agg_pnl_df.columns = pd.MultiIndex.from_tuples(
        [(time_period, 'P&L') for time_period in n_monthly_obs_mapping.keys()])
    portfolio_asset_agg_time_weighted_returns_df.columns = pd.MultiIndex.from_tuples(
        [(time_period, 'Return') for time_period in n_monthly_obs_mapping.keys()])
    portfolio_asset_agg_contribution_df.columns = pd.MultiIndex.from_tuples(
        [(time_period, 'Contribution(bps)') for time_period in n_monthly_obs_mapping.keys()])

    # Add NAV and NAV(%) column
    nav_df = pd.DataFrame(portfolio_asset_nav.iloc[-1].fillna(0))
    nav_df.columns = ['NAV']
    nav_df['%NAV'] = nav_df['NAV'] / nav_df['NAV'].sum()
    nav_df.loc['Total', :] = nav_df.sum(axis=0)

    # Format
    nav_df['NAV'] = nav_df['NAV'].apply(numeric_to_currency_str_format)
    nav_df['%NAV'] = nav_df['%NAV'].apply(numeric_to_percentage_str_format)

    nav_df.columns = pd.MultiIndex.from_tuples([(col, '') for col in nav_df.columns])

    # Join
    df = portfolio_asset_agg_pnl_df.join(portfolio_asset_agg_time_weighted_returns_df). \
        join(portfolio_asset_agg_contribution_df).join(nav_df)

    # Sort Columns and Rows
    contribution_column_tuples = [[(j, i) for i in ('P&L', 'Return', 'Contribution(bps)')]
                                  for j in n_monthly_obs_mapping.keys()]
    contribution_column_tuples = [col for sublist in contribution_column_tuples for col in sublist]
    column_tuples = nav_df.columns.to_list() + contribution_column_tuples
    df = df.loc[custom_sort_products + ['Total'], pd.MultiIndex.from_tuples(column_tuples)]

    # Add Index to the Columns
    df = df.rename_axis(pd.MultiIndex.from_tuples([('Product', '')]))
    df = df.reset_index(drop=False)

    return dbc.Table.from_dataframe(
        df,
        striped=True,
        bordered=True,
        hover=True,
        responsive='sm',
        style={'textAlign': 'center', 'position': 'sticky', 'white-space': 'nowrap', 'font-size': 'x-small',
               'font-family': 'Segoe UI'}
    )


def cumulative_pnl_graph():
    pnl_df = db.load_portfolio_asset_pnl().loc[:, custom_sort_products]
    pnl_df = pnl_df.cumsum().melt(ignore_index=False, var_name='Product', value_name='Cumulative P&L(€)')
    return pa.area_plot(pnl_df.reset_index(drop=False), x='date', y='Cumulative P&L(€)', color='Product',
                        yaxis_tick_format=',.0f', title='Cumulative P&L Attribution')


def risk_attribution_graph():
    current_weights = db.load_portfolio_asset_nav(groupby_field='product_id').loc[end_date]
    current_weights = current_weights / current_weights.sum()
    product_return_series = db.load_product_return_time_series(mapping_field='product_id',
                                                               frequency='1M').loc[:, current_weights.index].dropna()
    risk_attribution_df = pa.risk_attribution(pa.annualize_vol, product_return_series, current_weights.values)
    risk_attribution_df = risk_attribution_df.reset_index(drop=False)
    risk_attribution_df.columns = ['product_id', 'Volatility Attribution']
    mapping = db.generate_product_id_to_field_dict(mapping_field='name')
    risk_attribution_df['Product'] = risk_attribution_df['product_id'].apply(lambda idd: mapping[idd])
    risk_attribution_df = risk_attribution_df.groupby(['Product']). \
        agg({'Volatility Attribution': 'sum'})
    plot_series = (risk_attribution_df['Volatility Attribution'] * 10000)
    return pa.pie_chart(plot_series[custom_sort_products].round(0).copy(deep=True),
                        title='Volatility Attribution (bps, using current portfolio weights)',
                        show_legend=False, percentage=False, sort=False)


# App Instantiation
app = dash.Dash(name=__name__, title='DEGIRO - Portfolio Analytics',
                suppress_callback_exceptions=True)

navbar = dbc.NavbarSimple(
    brand='DEGIRO - Portfolio Analytics',
    brand_style={'fontSize': '150%'},
    brand_href='',
    color=plt.COLORS['blue'],
    dark=True
)

portfolio_overview_controls = [
    dbc.Row([
        dbc.Col([
            html.P('Snapshot Date:', style=TEXT_STYLE),
            dcc.DatePickerSingle(
                id='portfolio_overview_date_picker',
                min_date_allowed=start_date,
                max_date_allowed=end_date,
                initial_visible_month=end_date,
                display_format='YYYY-MM-DD',
                date=end_date,
                style=TEXT_STYLE
            )
        ]),
        dbc.Col([
            html.P('Group by:', style=TEXT_STYLE),
            dcc.Dropdown(options=[{'label': GROUPBY_FIELD_TO_LABEL[key], 'value': key}
                                  for key in GROUPBY_FIELD_TO_LABEL.keys()],
                         placeholder='Group by...',
                         value='name',
                         multi=False,
                         id='portfolio_overview_groupby_dropdown',
                         style=TEXT_STYLE)
        ]),
        dbc.Col([
            html.P('Show Chart in %:', style=TEXT_STYLE),
            BooleanSwitch(id='portfolio_overview_percentage_switch', on=True)
        ])
    ])
]

portfolio_nav_growth_controls = [
    html.P('Date Range:', style=TEXT_STYLE),
    dcc.DatePickerRange(
        id='portfolio_nav_growth_date_picker',
        min_date_allowed=start_date,
        max_date_allowed=end_date,
        initial_visible_month=end_date,
        display_format='YYYY-MM-DD',
        start_date=start_date,
        end_date=end_date,
        style=TEXT_STYLE
    ),
    html.Br(),
    html.Br(),
    html.P('Frequency:', style=TEXT_STYLE),
    dcc.Dropdown(options=[{'label': FREQUENCY_TO_LABEL[key], 'value': key}
                          for key in FREQUENCY_TO_LABEL.keys()],
                 placeholder='Frequency...',
                 value='1Y',
                 multi=False,
                 id='portfolio_nav_growth_frequency_dropdown',
                 style=TEXT_STYLE),
    html.Br(),
    html.P('Show Charts in %:', style=TEXT_STYLE),
    BooleanSwitch(id='portfolio_nav_growth_percentage_switch', on=False)
]

benchmarking_controls = [
    html.P('Date Range:', style=TEXT_STYLE),
    dcc.DatePickerRange(
        id='benchmarking_date_picker',
        min_date_allowed=start_date_monthly,
        max_date_allowed=end_date,
        initial_visible_month=end_date_monthly,
        display_format='YYYY-MM-DD',
        start_date=start_date_monthly,
        end_date=end_date_monthly,
        style=TEXT_STYLE
    ),
    html.Br(),
    html.Br(),
    html.P('Compare with Products:', style=TEXT_STYLE),
    dcc.Dropdown(options=[{'label': value, 'value': value}
                          for value in product_returns.columns],
                 placeholder='Select one or multiple Products',
                 multi=True,
                 id='benchmarking_products_dropdown',
                 style=TEXT_STYLE),
    html.Br(),
    html.P('Compare with Indices:', style=TEXT_STYLE),
    dcc.Dropdown(options=[{'label': value, 'value': value}
                          for value in market_series_returns.columns],
                 placeholder='Select one or multiple Indices',
                 multi=True,
                 id='benchmarking_indices_dropdown',
                 style=TEXT_STYLE),
    html.Br(),
    html.P("Indices' Currency:", style=TEXT_STYLE),
    dcc.Dropdown(options=[{'label': value, 'value': value}
                          for value in ('Local', 'EUR', 'USD', 'JPY')],
                 placeholder='Select a currency...',
                 multi=False,
                 id='benchmarking_currency_dropdown',
                 value='Local',
                 style=TEXT_STYLE),
]

performance_analytics_controls = [
    html.P('Date Range:', style=TEXT_STYLE),
    dcc.DatePickerRange(
        id='performance_analytics_date_picker',
        min_date_allowed=start_date_monthly,
        max_date_allowed=end_date,
        initial_visible_month=end_date_monthly,
        display_format='YYYY-MM-DD',
        start_date=start_date_monthly,
        end_date=end_date_monthly,
        style=TEXT_STYLE
    ),
    html.Br(),
    html.Br(),
    html.P('Compare with Products:', style=TEXT_STYLE),
    dcc.Dropdown(options=[{'label': value, 'value': value}
                          for value in product_returns.columns],
                 placeholder='Select one or multiple Products',
                 multi=True,
                 id='performance_analytics_products_dropdown',
                 style=TEXT_STYLE),
    html.Br(),
    html.P('Compare with Indices:', style=TEXT_STYLE),
    dcc.Dropdown(options=[{'label': value, 'value': value}
                          for value in market_series_returns.columns],
                 placeholder='Select one or multiple Indices',
                 multi=True,
                 id='performance_analytics_indices_dropdown',
                 style=TEXT_STYLE),
    html.Br(),
    html.P("Indices' Currency:", style=TEXT_STYLE),
    dcc.Dropdown(options=[{'label': value, 'value': value}
                          for value in ('Local', 'EUR', 'USD', 'JPY')],
                 placeholder='Select a currency...',
                 multi=False,
                 id='performance_analytics_currency_dropdown',
                 value='Local',
                 style=TEXT_STYLE),
]

factor_exposure_controls = [
    html.P('Dependent variable:', style=TEXT_STYLE),
    dcc.Dropdown(options=[{'label': value, 'value': value}
                          for value in ['My Portfolio (current weights)', 'My Portfolio'] +
                          list(product_returns.columns)],
                 placeholder='Select the Dependent Variable',
                 multi=False,
                 value=['My Portfolio (current weights)'],
                 id='factor_exposure_dependent_variable_dropdown',
                 style=TEXT_STYLE),
    html.Br(),
    html.P('Factors:', style=TEXT_STYLE),
    dcc.Dropdown(options=[{'label': value, 'value': value}
                          for value in factor_series_returns.columns if value != 'RF'],
                 placeholder='Select one or multiple Factors',
                 value=['Market', 'Size', 'Value'],
                 multi=True,
                 id='factor_exposure_factors_dropdown',
                 style=TEXT_STYLE)
]
app.layout = dbc.Container(
    fluid=True,
    children=[
        navbar,
        html.Br(),
        dbc.Tabs([
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            html.Br()
                        ]),
                        dbc.Card([
                            dbc.Row([
                                dbc.Col([dbc.Card(portfolio_overview_controls, body=True)], width=12),
                            ]),
                            dbc.Row([
                                html.Br()
                            ]),
                            dbc.Row(
                                id='nav_snapshot_table'
                            ),
                        ], body=True, style={'height': '100%', 'overflow': 'scroll'})
                    ], width=6),

                    dbc.Col([
                        dbc.Row([
                            html.Br()
                        ]),

                        dbc.Card(
                            dbc.Row([
                                dcc.Graph(id='nav_snapshot_graph')
                            ]), body=True, style={'height': '100%'})
                    ], width=6),
                ]),
                dbc.Row([html.Br(), html.Br()])
            ], label='Overview'),
            dbc.Tab([
                dbc.Row([
                    html.Br()
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([dbc.Row(id='portfolio_nav_growth_table_0'),
                                  dbc.Row(id='portfolio_nav_growth_table_1')], body=True, style={'height': '100%'})
                    ], width=4),
                    dbc.Col([
                        dbc.Card(dcc.Graph(id='portfolio_nav_growth_graph'), body=True, style={'height': '100%'})
                    ], width=8)
                ]),
                dbc.Row([
                    html.Br()
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(portfolio_nav_growth_controls, body=True, style={'height': '100%'})
                    ], width=2),
                    dbc.Col([
                        dbc.Card(dcc.Graph(id='portfolio_nav_growth_comparison_graph'),
                                 body=True, style={'height': '100%'})
                    ], width=5),
                    dbc.Col([
                        dbc.Card(dcc.Graph(id='portfolio_nav_growth_comparison_pie_graph'),
                                 body=True, style={'height': '100%'})
                    ], width=5)
                ]),
                dbc.Row(html.Br())
            ], label='Capital Growth'),
            dbc.Tab([
                dbc.Row([
                    html.Br()
                ]),
                dbc.Card(
                    dbc.Row(
                        id='benchmarking_table'
                    ), body=True, style={'height': '100%', 'overflow': 'scroll'}),
                dbc.Row([
                    html.Br()
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(benchmarking_controls, body=True, style={'height': '100%'})
                    ], width=4),
                    dbc.Col([
                        dbc.Card(dcc.Graph(id='benchmarking_return_graph'),
                                 body=True, style={'height': '100%'})
                    ])
                ]),
                dbc.Row(html.Br())
            ], label='Performance'),
            dbc.Tab([
                dbc.Row([
                    html.Br()
                ]),
                dbc.Card(
                    dbc.Row(
                        id='performance_analytics_table'
                    ), body=True, style={'height': '100%', 'overflow': 'scroll'}),
                dbc.Row([
                    html.Br()
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(performance_analytics_controls, body=True, style={'height': '100%'})
                    ], width=4),
                    dbc.Col([
                        dbc.Card(dcc.Graph(id='performance_analytics_return_graph'),
                                 body=True, style={'height': '100%'})
                    ])
                ]),
                dbc.Row([
                    html.Br()
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(dcc.Graph(id='performance_analytics_volatility_graph'),
                                 body=True, style={'height': '100%'})
                    ], width=6),
                    dbc.Col([
                        dbc.Card(dcc.Graph(id='performance_analytics_drawdown_graph'),
                                 body=True, style={'height': '100%'})
                    ], width=6)
                ]),
                dbc.Row(html.Br())
            ], label='Risk'),
            dbc.Tab([
                dbc.Row([
                    html.Br()
                ]),
                dbc.Row([
                    dbc.Card([pnl_contribution_table()], body=True, style={'overflow': 'scroll'})
                ]),
                dbc.Row([
                    html.Br()
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(dcc.Graph(figure=cumulative_pnl_graph()), body=True, style={'height': '100%'})
                    ], width=6),
                    dbc.Col([
                        dbc.Card(dcc.Graph(figure=risk_attribution_graph()), body=True, style={'height': '100%'})
                    ], width=6),
                ]),
                dbc.Row(html.Br())
            ], label='Attribution'),
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            html.Br()
                        ]),
                        dbc.Card([
                            dbc.Row([
                                dbc.Col([dbc.Card(factor_exposure_controls, body=True)], width=12),
                            ]),
                            dbc.Row([
                                html.Br()
                            ]),
                            dbc.Row(
                                id='factor_exposure_table'
                            ),
                        ], body=True, style={'height': '100%', 'overflow': 'scroll'})
                    ], width=6),

                    dbc.Col([
                        dbc.Row([
                            html.Br()
                        ]),

                        dbc.Card(
                            dbc.Row([
                                dcc.Graph(id='factor_exposure_factor_loadings_graph')
                            ]), body=True, style={'height': '100%'})
                    ], width=6),
                ]),
                dbc.Row([html.Br(), html.Br()]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(dcc.Graph(id='factor_exposure_factor_correlation_graph'),
                                 body=True, style={'height': '100%'})
                    ], width=6),
                    dbc.Col([
                        dbc.Card(dcc.Graph(id='factor_exposure_volatility_attribution_graph'),
                                 body=True, style={'height': '100%'})
                    ], width=6),
                ]),
                dbc.Row(html.Br())
            ], label='Factor Loadings'),
        ])
    ]
)


# Callbacks
@app.callback(
    [Output('nav_snapshot_graph', 'figure'),
     Output('nav_snapshot_table', 'children')],
    [Input('portfolio_overview_date_picker', 'date'),
     Input('portfolio_overview_groupby_dropdown', 'value'),
     Input('portfolio_overview_percentage_switch', 'on')]
)
def nav_snapshot(snapshot_date: str, groupby_field: str, percentage: bool):
    data = db.load_portfolio_asset_nav(groupby_field=groupby_field).loc[snapshot_date]
    data = data.sort_values(ascending=False)
    title_addition = ' (%)' if percentage else ' (€)'
    title = 'Portfolio Snapshot as of ' + datetime.strptime(snapshot_date,
                                                            constants.DATE_FORMAT).strftime('%d %B %Y') + title_addition
    data_chart = data.rename('NAV')
    if not percentage:
        data_chart = data_chart.round(0)
    fig = pa.pie_chart(data_chart, percentage=percentage,
                       show_legend=False, title=title, hovertemplate_tick_format=',.0f',
                       percentage_precision=0)
    data_total = data.sum()
    data['Total'] = data_total
    data.rename_axis('Product', inplace=True)
    data = data.reset_index(drop=False)
    data.columns = [data.columns[0], 'NAV']
    data['%NAV'] = (data['NAV'] / data_total)

    # Formatting
    data['NAV'] = data['NAV'].apply(numeric_to_currency_str_format)
    data['%NAV'] = data['%NAV'].apply(numeric_to_percentage_str_format)
    return fig, \
           dbc.Table.from_dataframe(
               data,
               striped=True,
               bordered=True,
               hover=True,
               responsive='sm',
               style={'textAlign': 'center', 'position': 'sticky', 'white-space': 'nowrap', 'font-size': 'small',
                      'font-family': 'Segoe UI'}
           )


@app.callback(
    [Output('portfolio_nav_growth_graph', 'figure'),
     Output('portfolio_nav_growth_table_0', 'children'),
     Output('portfolio_nav_growth_table_1', 'children'),
     Output('portfolio_nav_growth_comparison_graph', 'figure'),
     Output('portfolio_nav_growth_comparison_pie_graph', 'figure')],
    [Input('portfolio_nav_growth_date_picker', 'start_date'),
     Input('portfolio_nav_growth_date_picker', 'end_date'),
     Input('portfolio_nav_growth_frequency_dropdown', 'value'),
     Input('portfolio_nav_growth_percentage_switch', 'on')]
)
def portfolio_growth(s_date: str, e_date: str, frequency: str, percentage: bool):
    data_0 = portfolio_nav.join(portfolio_cash_flows)
    data_0.columns = ['Portfolio NAV', 'Net Contributions']
    data_0['Net Contributions'] = -data_0['Net Contributions'].fillna(0)
    data_0['NAV Growth'] = data_0['Portfolio NAV'].diff()
    first_date = data_0.index.min()
    data_0.loc[first_date, 'NAV Growth'] = data_0.loc[first_date, 'Portfolio NAV']
    data_0['P&L'] = data_0['NAV Growth'] - data_0['Net Contributions']
    data_0['Cumulative Net Contributions'] = data_0['Net Contributions'].cumsum()
    data_0['Cumulative P&L'] = data_0['P&L'].cumsum()
    data_chart_0 = data_0[['Portfolio NAV', 'Cumulative Net Contributions', 'Cumulative P&L']]
    fig_0 = pa.time_series_plot(data_chart_0.loc[s_date:e_date], yaxis_tick_format=',.0f',
                                title='Portfolio NAV (€)'). \
        update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    columns = ['Net Contributions', 'P&L']
    data_0_table_input = data_0[columns].resample('1M').sum()

    total_l6m = data_0_table_input.iloc[-6:, :].sum().rename('L6M')
    average_l6m = data_0_table_input.iloc[-6:, :].mean().rename('L6M')
    total_l12m = data_0_table_input.iloc[-12:, :].sum().rename('L12M')
    average_l12m = data_0_table_input.iloc[-12:, :].mean().rename('L12M')
    total_l3y = data_0_table_input.iloc[-36:, :].sum().rename('L3Y')
    average_l3y = data_0_table_input.iloc[-36:, :].mean().rename('L3Y')
    total_itd = data_0_table_input.sum().rename('ITD')
    average_itd = data_0_table_input.mean().rename('ITD')
    data_0_table_0 = pd.concat([pd.DataFrame(df).T for df in
                                (total_l6m, total_l12m, total_l3y, total_itd)])
    data_0_table_0['Total'] = data_0_table_0[columns].sum(axis=1)
    data_0_table_1 = pd.concat([pd.DataFrame(df).T for df in
                                (average_l6m, average_l12m, average_l3y, average_itd)])
    data_0_table_1['Total'] = data_0_table_1[columns].sum(axis=1)

    data_0_table_0 = data_0_table_0.applymap(numeric_to_currency_str_format)
    data_0_table_0.columns = pd.MultiIndex.from_tuples([('Sum', col) for col in columns + ['Total']])
    data_0_table_0 = data_0_table_0.rename_axis('Time-Period')
    data_0_table_0 = data_0_table_0.reset_index(drop=False)

    data_0_table_1 = data_0_table_1.applymap(numeric_to_currency_str_format)
    data_0_table_1.columns = pd.MultiIndex.from_tuples([('Monthly Average', col) for col in columns + ['Total']])
    data_0_table_1 = data_0_table_1.rename_axis('Time-Period')
    data_0_table_1 = data_0_table_1.reset_index(drop=False)

    table_0 = dbc.Table.from_dataframe(
        data_0_table_0,
        striped=True,
        bordered=True,
        hover=True,
        responsive='sm',
        style={'textAlign': 'center', 'position': 'sticky', 'white-space': 'nowrap', 'font-size': 'small',
               'font-family': 'Segoe UI'})

    table_1 = dbc.Table.from_dataframe(
        data_0_table_1,
        striped=True,
        bordered=True,
        hover=True,
        responsive='sm',
        style={'textAlign': 'center', 'position': 'sticky', 'white-space': 'nowrap', 'font-size': 'small',
               'font-family': 'Segoe UI'})

    data_1 = portfolio_nav.loc[s_date:e_date].resample(frequency).last() \
        .join(portfolio_cash_flows.loc[s_date:e_date].resample(frequency).sum())
    data_1.columns = ['Ending NAV', 'Net Contributions']
    data_1['NAV Growth'] = data_1['Ending NAV'].diff()
    first_date = data_1.index.min()
    data_1.loc[first_date, 'NAV Growth'] = data_1.loc[first_date, 'Ending NAV']
    data_1['Net Contributions'] = -data_1['Net Contributions']
    data_1['P&L'] = data_1['NAV Growth'] - data_1['Net Contributions']
    data_1 = data_1[['P&L', 'Net Contributions']].melt(ignore_index=False, var_name='Growth Source',
                                                       value_name='NAV Growth(€)')
    data_1 = data_1.reset_index(drop=False)
    format_mapping = {'1Y': '%Y', '1M': '%b %Y', '1D': constants.DATE_FORMAT}
    data_1['date'] = data_1['date'].apply(lambda dt: dt.strftime(format_mapping[frequency]))
    if percentage:
        data_1['NAV Growth Abs.(€)'] = data_1['NAV Growth(€)'].abs()
        data_1['NAV Growth Total(€)'] = data_1.groupby(['date'])['NAV Growth Abs.(€)'].transform('sum')
        data_1['NAV Growth(%)'] = data_1['NAV Growth Abs.(€)'] / data_1['NAV Growth Total(€)']
    fig_1_yaxis_tick_format = '.1%' if percentage else ',.0f'
    fig_1_2_unit_of_measure = '%' if percentage else '€'
    fig_1 = pa.bar_grouped_plot(data_1, x='date', y=f'NAV Growth({fig_1_2_unit_of_measure})', color='Growth Source',
                                yaxis_tick_format=fig_1_yaxis_tick_format,
                                title=f'P&L vs. Net Contributions ({fig_1_2_unit_of_measure})',
                                barmode='relative', show_legend_at_bottom=True)

    data_2 = data_1.groupby(['Growth Source']).agg({'NAV Growth(€)': 'sum'})['NAV Growth(€)']
    if not percentage:
        data_2 = data_2.round(0)
    fig_2 = pa.pie_chart(data_2,
                         title=f'P&L vs. Net Contributions - Full Period ({fig_1_2_unit_of_measure})',
                         percentage=percentage,
                         hovertemplate_tick_format=',.0f',
                         percentage_precision=0
                         )
    return fig_0, table_0, table_1, fig_1, fig_2


@app.callback(
    [Output('benchmarking_table', 'children'),
     Output('benchmarking_return_graph', 'figure')
     ],
    [Input('benchmarking_date_picker', 'start_date'),
     Input('benchmarking_date_picker', 'end_date'),
     Input('benchmarking_products_dropdown', 'value'),
     Input('benchmarking_indices_dropdown', 'value'),
     Input('benchmarking_currency_dropdown', 'value')
     ]
)
def benchmarking(s_date: str, e_date: str, product_names: str, index_names: str, currency: str):
    product_returns_adj = db.load_product_return_time_series(frequency='1M')
    market_series_returns_adj = db.load_market_return_series(frequency='1M', currency=currency)
    data = pd.DataFrame(portfolio_returns.rename('My Portfolio'))
    if product_names is not None:
        data = data.join(product_returns_adj[product_names])
    if index_names is not None:
        data = data.join(market_series_returns_adj[index_names])

    data = data.dropna(how='all')

    data_table = pa.return_stats(data).rename_axis('Series')
    fig = pa.cumulative_returns_plot(data.loc[s_date:e_date])

    # Formatting

    columns = ['MTD Returns', 'QTD Returns', 'YTD Returns', 'L12M Returns',
               'L12M Volatility', 'L3Y Returns', 'L3Y Volatility', 'ITD Returns',
               'ITD Volatility']
    columns_tuples = [('MTD', 'Return'), ('QTD', 'Return'), ('YTD', 'Return'), ('L12M', 'Return'),
                      ('L12M', 'Volatility'), ('L3Y', 'Return'), ('L3Y', 'Volatility'), ('ITD', 'Return'),
                      ('ITD', 'Volatility')]
    columns_mapping = dict(zip(columns, columns_tuples))

    data_table = data_table.applymap(numeric_to_percentage_str_format)
    data_table.columns = pd.MultiIndex.from_tuples([columns_mapping[col] for col in data_table.columns])

    table = dbc.Table.from_dataframe(
        data_table.reset_index(drop=False),
        striped=True,
        bordered=True,
        hover=True,
        responsive='sm',
        style={'textAlign': 'center', 'position': 'sticky', 'white-space': 'nowrap', 'font-size': 'small',
               'font-family': 'Segoe UI'})

    return table, fig


@app.callback(
    [Output('performance_analytics_return_graph', 'figure'),
     Output('performance_analytics_drawdown_graph', 'figure'),
     Output('performance_analytics_volatility_graph', 'figure'),
     Output('performance_analytics_table', 'children')
     ],
    [Input('performance_analytics_date_picker', 'start_date'),
     Input('performance_analytics_date_picker', 'end_date'),
     Input('performance_analytics_products_dropdown', 'value'),
     Input('performance_analytics_indices_dropdown', 'value'),
     Input('performance_analytics_currency_dropdown', 'value')
     ]
)
def performance_analytics(s_date: str, e_date: str, product_names: str, index_names: str, currency: str):
    product_returns_adj = db.load_product_return_time_series(frequency='1M')
    market_series_returns_adj = db.load_market_return_series(frequency='1M', currency=currency)
    data = pd.DataFrame(portfolio_returns.rename('My Portfolio'))
    if product_names is not None:
        data = data.join(product_returns_adj[product_names])
    if index_names is not None:
        data = data.join(market_series_returns_adj[index_names])
    data = data.dropna(how='all').loc[s_date:e_date]

    fig_0 = pa.returns_distribution_plot(data, marginal='box', title='Monthly Returns Distribution')
    fig_1 = pa.drawdown_comparison_plot(data)
    temp = data.dropna().shape[0]
    temp = 12 if temp < 36 else temp
    rolling_factor = min(36, temp)
    fig_2 = pa.rolling_volatility_plot(data, rolling_factor=rolling_factor)

    performance_stats_df = pa.performance_stats(data).rename_axis('Series').reset_index(drop=False)

    # Formatting
    int_columns = ['Longest Drawdown(Months)']
    pct_columns = ['Return (Ann.)', 'Volatility (Ann.)', 'VaR 95%', 'CVaR 95%', 'Max Drawdown', 'Up Months',
                   'Down Months', 'Largest Positive Month', 'Largest Negative Month']
    float_columns = ['Skewness', 'Kurtosis']
    performance_stats_df[int_columns] = performance_stats_df[int_columns].applymap(numeric_to_int_str_format)
    performance_stats_df[pct_columns] = performance_stats_df[pct_columns].applymap(numeric_to_percentage_str_format)
    performance_stats_df[float_columns] = performance_stats_df[float_columns].applymap(numeric_to_float_str_format)

    table = dbc.Table.from_dataframe(
        performance_stats_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive='sm',
        style={'textAlign': 'center', 'position': 'sticky', 'white-space': 'nowrap', 'font-size': 'small',
               'font-family': 'Segoe UI'})

    return fig_0, fig_1, fig_2, table


@app.callback(
    [Output('factor_exposure_table', 'children'),
     Output('factor_exposure_factor_loadings_graph', 'figure'),
     Output('factor_exposure_factor_correlation_graph', 'figure'),
     Output('factor_exposure_volatility_attribution_graph', 'figure')
     ],
    [Input('factor_exposure_dependent_variable_dropdown', 'value'),
     Input('factor_exposure_factors_dropdown', 'value'),
     ]
)
def factor_exposure(dependent_variable_name: str, factor_names: str):
    # Extract factors
    factors = factor_series_returns[factor_names]

    # Construct Portfolio at Current Weights
    current_weights = db.load_portfolio_asset_nav(groupby_field='product_id').loc[end_date]
    current_weights = current_weights / current_weights.sum()
    product_return_series = db.load_product_return_time_series(mapping_field='product_id',
                                                               frequency='1M').loc[:, current_weights.index]. \
        dropna()
    my_port_current_w = product_return_series.multiply(current_weights.values, axis=1).sum(axis=1)
    my_port_current_w.name = 'My Portfolio (current weights)'
    data = pd.DataFrame(my_port_current_w)

    my_port = portfolio_returns.copy(deep=True).rename('My Portfolio')
    data = data.join(my_port).join(product_returns)

    dependent_variable = data[dependent_variable_name]

    # Run Regression and format table
    table_data = pa.regression(dependent_variable, factors)
    table_data.columns = ['Coefficient', 'Std. Error', 'T-Stat', 'p-value']
    table_data[['Coefficient', 'Std. Error', 'T-Stat']] = table_data[['Coefficient', 'Std. Error', 'T-Stat']].applymap(
        lambda x: numeric_to_float_str_format(x, precision=2))
    table_data['p-value'] = table_data['p-value'].apply(lambda x: numeric_to_float_str_format(x, precision=4))

    table = dbc.Table.from_dataframe(
        table_data.rename_axis('Factor').reset_index(drop=False),
        striped=True,
        bordered=True,
        hover=True,
        responsive='sm',
        style={'textAlign': 'center', 'position': 'sticky', 'white-space': 'nowrap', 'font-size': 'small',
               'font-family': 'Segoe UI'})

    dependent_variable_name = dependent_variable_name[0] if isinstance(dependent_variable_name, list) else\
        dependent_variable_name
    factor_loadings = table_data['Coefficient'].rename('Factor Loadings')
    fig_0 = pa.horizontal_bar_plot(factor_loadings.iloc[1:], title=f'{dependent_variable_name} - Factor Loadings')

    # Correlations
    fig_1 = pa.correlation_plot(pd.DataFrame(dependent_variable).join(factors), title='Correlations')

    # Volatility Attribution
    fig_2 = pa.horizontal_waterfall_plot(pa.factor_volatility_attribution(dependent_variable, factors),
                                         title=f'{dependent_variable_name} - Factor Contributions to Risk')

    return table, fig_0, fig_1, fig_2


if __name__ == '__main__':
    app.run_server(debug=False)
