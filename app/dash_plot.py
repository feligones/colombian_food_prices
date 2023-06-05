
import os
import dash
from dash import html
import plotly.graph_objects as go
from dash import dcc
import plotly.express as px
from dash.dependencies import Input, Output
import boto3
from dotenv import load_dotenv
import pandas as pd

# Importing custom modules
from conf import utils as uts
from conf import settings as sts

# Load ENV secrets
# assert load_dotenv(), "Environment file couldnt load"
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

# Create an S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Load the dataset from S3
objects = s3_client.list_objects_v2(Bucket=AWS_BUCKET_NAME, Prefix = sts.S3_PROJECT_PATH)
pqt_objects = [obj for obj in objects['Contents'] if obj['Key'].endswith('.parquet')]
last_pqt_object = sorted(pqt_objects, key=lambda x: x['LastModified'], reverse=True)[0]

s3_client.download_file(Filename = 'data.parquet', Bucket=AWS_BUCKET_NAME, Key=last_pqt_object['Key'])

df = pd.read_parquet('data.parquet')

os.remove('data.parquet')

# Create a Dash application
app = dash.Dash(__name__)

# Get unique markets and products in the dataset
unique_markets = df['market'].unique()
unique_products = df['product'].unique()

most_important_markets = [
    'medellin central mayorista de antioquia', 
    'bogota dc corabastos',
    'cucuta cenabastos',
    'barranquilla barranquillita'
    ]

# Define the layout of the web page
app.layout = html.Div(
    style={
        'font-family': 'Arial, sans-serif',
        'padding': '30px',
        'background-color': '#f8f8f8'
    },
    children=[
        html.H1(
            'Termómetro de Precios en las Centrales de Abasto de Colombia',
            style={
                'text-align': 'center',
                'color': '#333333',
                'font-size': '32px',
                'margin-bottom': '30px',
                'text-transform': 'uppercase',
                'letter-spacing': '2px'
            }
        ),

        html.Div(
            style={
                'text-align': 'center',
                'margin-bottom': '20px'
            },
            children=[
                html.H2(
                    'Central de Abasto',
                    style={'color': '#333333', 'font-size': '24px', 'margin-bottom': '10px'}
                ),
                dcc.Dropdown(
                    id='market-dpdn',
                    options=[{'label': s.upper(), 'value': s} for s in sorted(unique_markets)],
                    value=most_important_markets[0],
                    clearable=False,
                    style={'font-size': '16px'}
                )
            ]
        ),

        html.Div(
            style={
                'text-align': 'center',
                'margin-bottom': '50px'
            },
            children=[
                html.H2(
                    'Producto',
                    style={'color': '#333333', 'font-size': '24px', 'margin-bottom': '10px'}
                ),
                dcc.Dropdown(
                    id='product-dpdn',
                    options=[],
                    value='huevo rojo aa',
                    clearable=False,
                    style={ 'font-size': '16px'}
                )
            ]
        ),

        html.Div(
            id='graph-container-1',
            style={'margin-bottom': '30px'}
        ),
        html.Div(
            id='graph-container-2',
            style={'margin-bottom': '30px'}
        ),
        html.Div(
            id='graph-container-3',
            style={'margin-bottom': '30px'}
        ),
        html.Div(
            id='graph-container-4'
        )
    ]
)


# Define a callback function to update the product dropdown menu based on the selected market
@app.callback(
    Output('product-dpdn', 'options'),
    Output('product-dpdn', 'value'),
    Input('market-dpdn', 'value'),
)
def set_product_options(chosen_market):
    # Filter the dataset by the chosen market
    dff = df[df['market'] == chosen_market]
    # Get unique products available in the filtered dataset
    products_of_market = [{'label': c.upper(), 'value': c} for c in sorted(dff['product'].unique())]
    # Set the default product selection to the first available product
    values_selected = products_of_market[0]['value']
    # Return the options for the product dropdown menu and the default selection
    return products_of_market, values_selected


# Define a callback function to update the graph based on the selected market and product
@app.callback(
    Output('graph-container-1', 'children'),
    Output('graph-container-2', 'children'),
    Output('graph-container-3', 'children'),
    Output('graph-container-4', 'children'),
    Input('market-dpdn', 'value'),
    Input('product-dpdn', 'value'),
    prevent_initial_call=True
)
def graph_update(market_value, product_value):
    # Filter the dataset by the selected market and product
    filter_df = df.loc[
        (df['product'] == product_value) &
        (df['market'] == market_value)
    ].copy()
    filter_df.sort_values('date', inplace=True)
    filter_df['mean_price_diff_m'] = 100 * filter_df['mean_price'].diff(1) / filter_df['mean_price'].shift(1)
    filter_df['mean_price_diff_y'] = 100 * filter_df['mean_price'].diff(12) / filter_df['mean_price'].shift(12)
    filter_df['color_m'] = filter_df['mean_price_diff_m'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    filter_df['color_y'] = filter_df['mean_price_diff_y'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')

    # Define the color mapping
    color_map = {'Positive': '#00b140', 'Negative': '#ff5050'}

    # Create a line plot using Plotly Express
    fig_level = px.line(filter_df, x='date', y='mean_price', markers=True)
    # Set the title and axis labels of the plot
    fig_level.update_layout(
        title=f'Nivel de Precios de {product_value.upper()} en {market_value.upper()}',
        xaxis_title='Fecha',
        yaxis_title='Precio $ / Kg',
        template='plotly_white'
    )

    fig_change_m = px.bar(filter_df, x='date', y='mean_price_diff_m', color='color_m', color_discrete_map=color_map)
    fig_change_m.update_layout(
        title=f'Cambio % Mensual {product_value.upper()} en {market_value.upper()}',
        xaxis_title='Fecha',
        yaxis_title='Cambio Precio % M/M',
        showlegend=False,
        template='plotly_white'
    )

    fig_change_y = px.bar(filter_df, x='date', y='mean_price_diff_y', color='color_y', color_discrete_map=color_map)
    fig_change_y.update_layout(
        title=f'Cambio % Anual {product_value.upper()} en {market_value.upper()}',
        xaxis_title='Fecha',
        yaxis_title='Cambio Precio % Y/Y',
        showlegend=False,
        template='plotly_white'
    )

    # Create a line plot using Plotly Express for multiple markets
    multi_market_df = df.loc[
        (df['product'] == product_value) & 
        (df['market'].isin(most_important_markets + [market_value]))
        ].copy()
    multi_market_df.sort_values(['market', 'date'], inplace=True)
    fig_multi_market = px.line(multi_market_df, x='date', y='mean_price', color='market', markers=True)
    fig_multi_market.update_layout(
        title=f'Comportamiento del Precio de {product_value.upper()} en las Centrales de Abasto de Colombia',
        xaxis_title='Fecha',
        yaxis_title='Precio $',
        plot_bgcolor='#f8f8f8'
    )

    # Return a Plotly graph object
    return (
        dcc.Graph(figure=fig_level),
        dcc.Graph(figure=fig_change_m),
        dcc.Graph(figure=fig_change_y),
        dcc.Graph(figure=fig_multi_market),
    )

# Make the app callable
app = app.server 

# Run the Dash application
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
