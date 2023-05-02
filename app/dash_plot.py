import dash
import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Input, Output
from conf import utils as uts
from conf import settings as sts

app = dash.Dash()

df = uts.load_artifact('prices_dataframe', sts.LOCAL_ARTIFACTS_PATH)

unique_markets = df['market'].unique()

unique_products = df['product'].unique()

# app.layout = html.Div(id = 'parent', children = [
#     html.H1(id = 'H1', children = 'Precios de Alimentos en las Centrales de Abasto de Colombia', style = {'textAlign':'center',\
#                                             'marginTop':40,'marginBottom':40}),
# 
#         dcc.Dropdown( id = 'dropdown',
#         options = unique_products,
#         value = 'Huevo rojo AA'),
#         dcc.Graph(id = 'line_plot')
#     ])

app.layout = html.Div([
    html.Label("Central de Abasto:", style={'fontSize':30, 'textAlign':'center'}),
    dcc.Dropdown(
        id='market-dpdn',
        options=[{'label': s, 'value': s} for s in sorted(unique_markets)],
        value='Medellín, Central Mayorista de Antioquia',
        clearable=False
    ),
    html.Label("Productos:", style={'fontSize':30, 'textAlign':'center'}),
    dcc.Dropdown(id='product-dpdn',
                 options=[],
                 value = ['Huevo rojo AA']),
    html.Div(id='graph-container', children=[])
])

@app.callback(
    Output('product-dpdn', 'options'),
    Output('product-dpdn', 'value'),
    Input('market-dpdn', 'value'),
)

def set_product_options(chosen_market):
    dff = df[df['market']==chosen_market]
    products_of_market = [{'label': c, 'value': c} for c in sorted(dff['product'].unique())]
    values_selected = [x['value'] for x in products_of_market]
    return products_of_market, values_selected


@app.callback(
    Output('graph-container', 'children'),
    Input('market-dpdn', 'value'),
    Input('product-dpdn', 'value'),
    prevent_initial_call=True
)

def graph_update(market_value, product_value):
    # print(market_value, product_value)
    filter_df = df.loc[
        (df['product'] == product_value)&
        (df['market'] == market_value)
        ].copy()
    fig = px.line(filter_df, x = 'date', y = 'mean_price', markers = True)
    
    fig.update_layout(title = f'{product_value} IN {market_value}',
                      xaxis_title = 'Fecha',
                      yaxis_title = 'Precio $'
                      )
    return dcc.Graph(id='display-map', figure=fig)  



if __name__ == '__main__': 
    app.run_server(debug=True)