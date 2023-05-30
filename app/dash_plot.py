import dash
from dash import html
import plotly.graph_objects as go
from dash import dcc
import plotly.express as px
from dash.dependencies import Input, Output

# Importing custom modules
from conf import utils as uts
from conf import settings as sts

# Create a Dash application
app = dash.Dash()

# Load the dataset from a saved artifact
df = uts.load_artifact('prices_dataframe', sts.LOCAL_ARTIFACTS_PATH)

# Get unique markets and products in the dataset
unique_markets = df['market'].unique()
unique_products = df['product'].unique()

# Define the layout of the web page
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

# Define a callback function to update the product dropdown menu based on the selected market
@app.callback(
    Output('product-dpdn', 'options'),
    Output('product-dpdn', 'value'),
    Input('market-dpdn', 'value'),
)
def set_product_options(chosen_market):
    # Filter the dataset by the chosen market
    dff = df[df['market']==chosen_market]
    # Get unique products available in the filtered dataset
    products_of_market = [{'label': c, 'value': c} for c in sorted(dff['product'].unique())]
    # Set the default product selection to the first available product
    values_selected = [x['value'] for x in products_of_market][0]
    # Return the options for the product dropdown menu and the default selection
    return products_of_market, values_selected

# Define a callback function to update the graph based on the selected market and product
@app.callback(
    Output('graph-container', 'children'),
    Input('market-dpdn', 'value'),
    Input('product-dpdn', 'value'),
    prevent_initial_call=Input('market-dpdn', 'value') != None and Input('product-dpdn', 'value') != None
)
def graph_update(market_value, product_value):
    # Filter the dataset by the selected market and product
    filter_df = df.loc[
        (df['product'] == product_value) &
        (df['market'] == market_value)
    ].copy()
    # Create a line plot using Plotly Express
    fig = px.line(filter_df, x='date', y='mean_price', markers=True)
    # Set the title and axis labels of the plot
    fig.update_layout(
        title=f'{product_value} IN {market_value}',
        xaxis_title='Fecha',
        yaxis_title='Precio $'
    )
    # Return a Plotly graph object
    return dcc.Graph(id='display-map', figure=fig)  

# Run the Dash application
if __name__ == '__main__': 
    app.run_server(debug=True)
