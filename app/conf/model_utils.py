
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.deterministic import DeterministicProcess

def make_lags(ts, lags):
    """
    This function computes the lagged values of a time series 
    and returns each lag as a feature column.
    
    Args:
        ts (pd.Series): time series to compute lags
        lags (int): number of lags to be computed.

    Returns:
        lags_df (pd.DataFrame): dataframe containing the lagged 
                                values of the input time series 
                                as columns.
    """
    lags_df = pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)

    return lags_df

def make_lags_features(ts, lags):
    simple_features = {
        f'y_lag_{i}': ts.shift(i)
        for i in range(1, lags + 1)
    }
    rolling_features = {
        f'y_rolling_std': ts.shift(1).rolling(lags).std()
    }
    
    return pd.concat(
        {
            **simple_features,
            **rolling_features
        }
        ,
        axis=1)

def get_X_det(y, forecast_steps = 1):
    res_robust = STL(y, period=12, robust=True).fit()
    f_t = max(0, 1 - res_robust.resid.var()/(res_robust.trend + res_robust.resid).var())
    f_s = max(0, 1 - res_robust.resid.var()/(res_robust.seasonal + res_robust.resid).var())
    
    dp = DeterministicProcess(
        index = y.index,
        constant = True,
        order = (2 if f_t > 0.5 else 0),
        drop = True,
        seasonal = (True if f_s > 0.5 else False)
    )
    
    X = dp.in_sample()
    X_fore = dp.out_of_sample(steps = forecast_steps)
    
    return X, X_fore

def get_model(y, X, model, return_residuals):
    model.fit(X, y)
    y_pred = pd.Series(model.predict(X), index = y.index)
    if return_residuals:
        residuals = y - y_pred
        return model, y_pred, residuals
    else:
        return model, y_pred

def get_X_res(residuals, lags):
    res_lags = make_lags_features(residuals, lags).dropna()
    res_lags_last = res_lags[-1:]
    res_lags_last.index = res_lags_last.index.shift(1)
    residuals_ri, res_lags = residuals.align(res_lags, join='inner', axis=0)
    return residuals_ri, res_lags_last, res_lags

def get_prediction(models, Xs, res_std):
    model_1 = models[0]
    model_2 = models[1]
    
    X_1 = Xs[0]
    X_2 = Xs[1]
    
    point_pred = model_1.predict(X_1)[0] + model_2.predict(X_2)[0]
    pred = {
        'point': point_pred,
        'lower': point_pred - res_std,
        'upper': point_pred + res_std,
        'index': X_1.index[0]
    }
    
    return pred

def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)
