
import pandas as pd
import numpy as np
from conf import settings as sts
from conf import utils as uts
from conf import model_utils as muts
from conf.mymodel import MyModel
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor


# Import Prices Dataset
prices_dataset = uts.load_artifact('prices_dataframe', sts.LOCAL_ARTIFACTS_PATH)
prices_dataset['date'] = pd.to_datetime(prices_dataset['date'])

# Set ID for each series (group-product-market)
prices_dataset['series_id'] = prices_dataset.groupby(['group', 'product', 'market']).ngroup()
get_date_range = lambda start: pd.date_range(start = start, end = prices_dataset['date'].max(), freq = 'MS')

# Get residuals and fitted Values from the deterministic model applied to each time series
multi_X_2 = []
multi_X_fore_2 = []
multi_y_2 = []
series_id_series = []
multi_y_pred_1 = []

predictions = {}

# For each TS:
for series_id, series_df in prices_dataset.groupby('series_id'):
    series_df.set_index('date', inplace=True)
    group = series_df['group'].unique()[0]
    # Extract, reindex and interpolate (lineal) price time series
    y = series_df['mean_price'].astype(float)
    date_range_idx = get_date_range(y.first_valid_index())
    y = y.reindex(date_range_idx).interpolate()
    
    # Get features for deterministic model
    X_1, X_fore_1 = muts.get_X_det(y, forecast_steps = 1)
    # Get model, fitted values and residuals from deterministic model
    model_1, y_pred_1, residuals = muts.get_model(y, X_1, LinearRegression(), return_residuals = True)
    # Save point forecast for deterministic model
    pred = {
        'point_det':model_1.predict(X_fore_1)[0]
    }
    predictions[series_id] = pred
    
    # Get response variable and features (lags) for residual model
    y_2, X_fore_2, X_2 = muts.get_X_res(residuals, lags=3)
    X_2['group'] = group
    X_fore_2['group'] = group

    # Add attributes to deterministic fitted values DF
    y_pred_frame_1 = y_pred_1.iloc[3:].to_frame()
    y_pred_frame_1['series_id'] = series_id
    y_pred_frame_1['y_true'] = y.iloc[3:]
    
    # Append all to DF lists
    multi_y_pred_1.append(y_pred_frame_1)
    multi_X_2.append(X_2)
    multi_X_fore_2.append(X_fore_2)
    multi_y_2.append(y_2)

# Concat lists to get dataframes for:
## Deterministic model inputs
multi_X_2 = pd.concat(multi_X_2)
multi_X_fore_2 = pd.concat(multi_X_fore_2)
## Deterministic model fitted values
multi_y_pred_1 = pd.concat(multi_y_pred_1)
## Residual model response variable
multi_y_2 = pd.concat(multi_y_2)

# Get forecast date
forecast_date = X_fore_1.index[0].strftime('%Y-%m')

# Build residual model inputs pipelines
feature_pipeline = ColumnTransformer(
    transformers=[
        ('group_onehot', OneHotEncoder(handle_unknown='ignore'), ['group']),
        ('numeric', 'passthrough', ['y_lag_1', 'y_lag_2', 'y_lag_3', 'y_rolling_std'])
    ],
)

# Fit feature pipeline and transform residual model inputs
feature_pipeline.fit(multi_X_2)
trans_multi_X_2 = feature_pipeline.transform(multi_X_2)

# Fit residual model and get fitted values
model_2, multi_y_pred_2 = muts.get_model(multi_y_2, trans_multi_X_2, KNeighborsRegressor(), return_residuals=False)

# Ensamble final model fitted values
multi_y_pred_1['y_pred'] = multi_y_pred_2 + multi_y_pred_1[0]

# Get final model residuals
multi_y_pred_1['residuals'] = multi_y_pred_1['y_true'] - multi_y_pred_1['y_pred']

# Compute each series residuals std
std_preds = multi_y_pred_1.groupby(['series_id'])['residuals'].std().to_dict()

# Get point forecast for residual model
point_res = pd.Series(
    model_2.predict(feature_pipeline.transform(multi_X_fore_2)), 
    index = np.arange(multi_X_fore_2.shape[0])
    ).to_dict()

# Compute final predictions (det + res model point forecasts) and forecast confidence intervals
final_predictions = {
    series_id:{
        'point_res': point_res[series_id],
        **predictions[series_id],
        'std': std_preds[series_id],
        'point': predictions[series_id]['point_det'] + point_res[series_id],
        'lower': predictions[series_id]['point_det'] + point_res[series_id] - 1.96*std_preds[series_id],
        'upper': predictions[series_id]['point_det'] + point_res[series_id] + 1.96*std_preds[series_id]
    }
    for series_id in prices_dataset['series_id'].unique()
}

# Build routing dicts
## tuple(product, market) -> series_id
series_to_id_dict = prices_dataset[['product', 'market', 'series_id']].drop_duplicates().set_index(['product', 'market']).squeeze().to_dict()
## series_id -> market
id_to_market_dict = {v:k[1] for k,v in series_to_id_dict.items()}
## product -> list(series_ids) product in different markets
product_to_series_dict = prices_dataset[['product', 'series_id']].drop_duplicates().groupby(['product'])['series_id'].apply(lambda x: list(np.unique(x))).to_dict()

routing_dicts = {
    'series_to_id_dict': series_to_id_dict,
    'id_to_market_dict': id_to_market_dict,
    'product_to_series_dict': product_to_series_dict
}

# Instance MyModel object
model = MyModel(final_predictions, routing_dicts, forecast_date)

# Save Model as artifact
uts.dump_artifact(model, 'model', sts.LOCAL_ARTIFACTS_PATH)