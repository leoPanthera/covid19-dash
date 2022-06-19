import os
import datetime as dt

import numpy as np
import pandas as pd

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px


class DataController:

    def __init__(self, data_dir='./data'):
        self.indicator_names = [
        #     'Community health workers (per 1,000 people)',
            ('Current health expenditure (% of GDP)', 'Health Cost / GDP'),
            ('Current health expenditure per capita (current US$)', 'Health Cost per capita'),
            ('Hospital beds (per 1,000 people)', 'Hospital Beds / 1k person'),
            ('Life expectancy at birth, total (years)', 'Life expectancy')
        ]
        self.country_code = pd.read_csv(os.path.join(data_dir, 'UID_ISO_FIPS_LookUp_Table.csv'), index_col='Combined_Key')
        self.covid_total, self.covid_ts = self.load_covid_data(data_dir)
        self.wdi_data = self.load_indicators(os.path.join(data_dir, 'WDI_data.csv'))
        self.wdi_indicators = self.select_indicators(self.indicator_names)
        self.total_with_indicators = self.postprocess()
#         self.total = self.get_period_summary_with_indicators(indicators)

    def load_covid_data(self, ts_data_dir):
        confirmed = pd.read_csv(os.path.join(ts_data_dir, 'time_series_covid19_confirmed_global.csv'))
        deaths = pd.read_csv(os.path.join(ts_data_dir, 'time_series_covid19_deaths_global.csv'))
        recovered = pd.read_csv(os.path.join(ts_data_dir, 'time_series_covid19_recovered_global.csv'))
        total, ts = None, None
        merge_col_ts = ['Country/Region', 'Date']
        for name, d in zip(['confirmed', 'deaths', 'recovered'], [confirmed, deaths, recovered]):
            total_, ts_ = self.get_period_summary(d, name=name)
            total = total_ if total is None else total.assign(**{name: total_[name]})
            ts = ts_ if ts is None else ts.assign(**{name: ts_[name]})
        ts['Date'] = ts['Date'].apply(lambda x: dt.date(int('20' + x.split('/')[2]), *map(int, x.split('/')[:2])))
        return total.merge(self.country_code, left_on='Country/Region', right_index=True), \
            ts.merge(self.country_code, left_on='Country/Region', right_index=True)

    def load_indicators(self, path):
        wdi_data = pd.read_csv(path)
#         wdi_data = wdi_data[list(wdi.columns[:4]) + ['2020']][['Country Name','Country Code', 'Indicator Name', '2020']]
        return wdi_data

    def get_period_summary(self, df, name, end=None):
        df = df.groupby('Country/Region').sum()
        df = df.drop(['Lat', 'Long'], axis=1)
        if end is None:
            end = len(df.columns)
        else:
            end = [i for i, d in enumerate(df.columns) if d == e]
            if len(end) == 1:
                end = end[0]
            else:
                end = len(df.columns)
        
        df['Country/Region'] = df.index
        melted = df.melt(
            id_vars=['Country/Region'],
            value_vars=df.columns[3:end],
            var_name='Date',
            value_name=name,
        )
        total = melted.groupby('Country/Region').agg({name: 'last'})
#         total['Country/Region'] = total.index
        return total, melted
    
    def select_indicators(self, indicators, end=None):
#             .merge(right=self.country_code, how='left', left_index=True, right_index=True)
        wdi = self.wdi_data
        indicator_data = None
        for indicator, rename in indicators:
            # rename = None
            selected = wdi[wdi['Indicator Name'] == indicator].set_index('Country Code')
            selected = selected.drop('Indicator Name', axis=1).drop('Country Name', axis=1).drop('Unnamed: 0', axis=1)
            rename = rename or indicator
            selected.columns = [rename]
            indicator_data = selected if indicator_data is None else indicator_data.assign(**{rename: selected[rename]})
        return indicator_data
    
    def merge_total_with_indicator(self, total, indicator):
        total = total.merge(right=indicator, how='left', left_on='iso3', right_index=True)
        return total

    def postprocess(self):
        total = self.merge_total_with_indicator(self.covid_total, self.wdi_indicators)
        total['death_per_million'] = total['deaths'] / total['Population'] * 1000000
        total['confirmed_per_million'] = total['confirmed'] / total['Population'] * 1000000
        total['death_rate'] = total['deaths'] / total['confirmed']
        total['sublinear_population'] = total['Population'] ** 0.66
        total = total[~total['Population'].isna()]
        return total

controller = DataController('./data')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dbc.Row([
        html.Div(style={'width': '5%', 'display': 'inline-block', 'padding': '0 20'}),
    ]),
    dbc.Row([
        html.Div(
            "Health Cost ( % GDP )", 
            style={'width': '15%', 'display': 'inline-block','text-align':'right', 'padding': '0 25 25 25'}
        ), 
        html.Div(dcc.RangeSlider(
            1, 18, 
            step=None, id='crossfilter-health-over-gdp',
            value=[
                controller.total_with_indicators['Health Cost / GDP'].min(),
                controller.total_with_indicators['Health Cost / GDP'].max() + 1
            ]), style={'width': '30%', 'display': 'inline-block'}),
        html.Div(
            "Health Cost per capita", 
            style={'width': '15%', 'display': 'inline-block','text-align':'right', 'padding': '0 25 25 25'}
        ), 
        html.Div(dcc.RangeSlider(
            10, 12000,
            step=None, id='crossfilter-health-per-capita',
            value=[
                controller.total_with_indicators['Health Cost per capita'].min()-1,
                controller.total_with_indicators['Health Cost per capita'].max()*1.1
            ]), style={'width': '30%', 'display': 'inline-block'})
    ]),
    dbc.Row([
        html.Div(style={'width': '20%', 'display': 'inline-block', 'padding': '0 20'}),
        html.Div(dcc.RadioItems(
            ['Confirmed Infection per million Person', 'Deaths per million Person'],
            'Confirmed Infection per million Person',
            id='y-axis',
            labelStyle={'display': 'inline-block', 'marginTop': '5px'}
        ), style={'width': '49%', 'padding': '0px 40px 20px 40px'}),
    ]),
    dbc.Row([
        html.Div([dcc.Graph(
            id='total-confirmed-per-million',
            hoverData={'points': [{'customdata': 'US'}]},
            style={'height': '40vh'}
        )], style={'width': '46%', 'display': 'inline-block', 'padding': '0 20', }), 
        # html.Div(style={'width': '3%', 'display': 'inline-block', 'padding': '0 20'}),
        html.Div([dcc.Graph(
            id='total-death-per-million',
            hoverData={'points': [{'customdata': 'US'}]},
            style={'height': '40vh'}
        )], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20', })
    ]),
    # dbc.Row([
    #     html.Div(dcc.Slider(
    #         0, (controller.covid_ts['Date'].max() - controller.covid_ts['Date'].min()).days, 
    #         step=None, id='crossfilter-datetime',
    #         value=(controller.covid_ts['Date'].max() - controller.covid_ts['Date'].min()).days,
    #         marks={int(i):str(j) for i, j in zip(range((controller.covid_ts['Date'].max() - controller.covid_ts['Date'].min()).days),
    #                                             controller.covid_ts['Date'].unique()) if i % 100 == 0},
    #     ), style={'width': '80%', 'display': 'inline-block', 'padding': '30 50 30 200'})
    # ]),
    dbc.Row([
        html.Div([dcc.Graph(
            id='ts-data',
            hoverData={'points': [{'customdata': 'US'}]}
        )], style={'width': '95%', 'display': 'inline-block', 'padding': '0 20'}), 
        # html.Div([dcc.Graph(
        #     id='total-death-per-million',
        #     hoverData={'points': [{'customdata': 'US'}]}
        # )], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'})
    ]),
], style={'width': '99%', 'padding': '30 50 30 50'})


@app.callback(
    Output('total-confirmed-per-million', 'figure'),
    Input('y-axis', 'value'),
    Input('crossfilter-health-per-capita', 'value'),
    Input('crossfilter-health-over-gdp', 'value'),)
def update_confirmed_graph(y_axis, health_pc, health_over_gdp):
    total = controller.total_with_indicators
    xname = 'Life expectancy' #'confirmed_affection_per_million', 
    yname = 'confirmed_per_million' if y_axis == 'Confirmed Infection per million Person' else 'death_per_million'
    color = 'death_rate'
    size = 'confirmed'
    total = total[total['death_rate'] < 0.06]
    total['transparency'] = np.zeros(total.shape[0])
    f = 1
    for n, r in zip(['Health Cost / GDP', 'Health Cost per capita'], [health_over_gdp, health_pc]):
        low, high = r
        s = total[n]
        f = f * ((s > low) & (s < high))
        # print(n, low, high)
    total['transparency'] = 1 - 0.9 * (1 - f)

    fig = px.scatter(total,
        x = xname,   
        y = yname,
        size = size,
        color = color,
        size_max = 20,
        opacity=total['transparency'],
        hover_data=[
            'Country_Region', 'confirmed', 'deaths', 'Population', 
            'Health Cost per capita',
            'Life expectancy'
        ]
    )
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    # fig.update_traces(customdata=total['iso3'])    
    fig.update(layout_coloraxis_showscale=False)
    return fig


@app.callback(
    Output('total-death-per-million', 'figure'),
    Input('crossfilter-health-per-capita', 'value'),
    Input('crossfilter-health-over-gdp', 'value'),)
def update_geo_graph(health_pc, health_over_gdp):
    total = controller.total_with_indicators
    xname = 'Life expectancy' #'confirmed_affection_per_million', 
    yname = 'death_per_million' #  y='confirmed_affection_per_million',
    color = 'death_rate'
    size = 'confirmed'
    total = total[total['death_rate'] < 0.06]
    f = 1
    for n, r in zip(['Health Cost / GDP', 'Health Cost per capita'], [health_over_gdp, health_pc]):
        low, high = r
        s = total[n]
        f = f * ((s > low) & (s < high))
    total['transparency'] = 1 - 0.9 * (1 - f)

    fig = px.scatter_geo(total,
        locations = "iso3",  
        size = size,
        color = color,
        size_max = 25,
        opacity=total['transparency'],
        hover_data=[
            'Country_Region', 'confirmed', 'deaths', 'Population', 
            'Health Cost per capita',
            'Life expectancy'
        ]
    )
    # fig.update_traces(customdata=total['iso3'])
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    return fig

# def update_death_graph(value, health_over_gdp):
#     total = controller.total_with_indicators
#     xname = 'Life expectancy' #'confirmed_affection_per_million', 
#     yname = 'death_per_million' #  y='confirmed_affection_per_million',
#     color = 'death_rate'
#     size = 'sublinear_population'
#     total = total[total['death_rate'] < 0.06]
#     fig = px.scatter(total,
#         x = xname,   
#         y = yname,
#         size = size,
#         color = color,
#         size_max = 20,
#         hover_data=[
#             'Country_Region', 'confirmed', 'deaths', 'Population', 
#             'Health Cost per capita',
#             'Life expectancy'
#         ]
#     )
#     # fig.update_traces(customdata=total['iso3'])
#     fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
#     return fig

# def _closure():
#     last = (None, None)

@app.callback(
    Output('ts-data', 'figure'),
    Input('total-confirmed-per-million', 'hoverData'))
def update_ts_graph(hoverData, ):
    country_name = hoverData['points'][0]['customdata'][0]
    ts_data = controller.covid_ts
    ts_data = ts_data[ts_data['Country_Region'] == country_name][['Date', 'confirmed', 'deaths']]
    ts_data['confirmed'] = ts_data['confirmed'].diff()
    ts_data['deaths'] = ts_data['deaths'].diff()
    # ts_data = ts_data[ts_data['Date'] < controller.covid_ts['Date'].min() + dt.timedelta(days=value)]
    fig = px.line(ts_data, x='Date', y=['confirmed', 'deaths'], range_y=[0, ts_data.confirmed.max() *1.01])
    fig.update_yaxes(title="confirmed affection in %s" % (country_name, ), )
    return fig
        

def update_world_map(self):
    pass


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')