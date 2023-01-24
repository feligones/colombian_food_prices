
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
    """
    This function computes the lagged values of a time series 
    and returns each lag as a feature column. Also it computes 
    rolling stats from the same lags as other features. In 
    this case, the std of all the lags.
    
    Args:
        ts (pd.Series): time series to compute lags
        lags (int): number of lags to be computed.

    Returns:
        lags_df (pd.DataFrame): dataframe containing the lagged 
                                values of the input time series 
                                and its rolling stats as columns.
    """
    simple_features = {
        f'y_lag_{i}': ts.shift(i)
        for i in range(1, lags + 1)
    }
    rolling_features = {
        'y_rolling_std': ts.shift(1).rolling(lags).std(),
        #'y_rolling_mean': ts.shift(1).rolling(lags).mean(),
    }
    
    lags_df = pd.concat(
        {
            **simple_features,
            **rolling_features
        }
        ,
        axis=1)
    
    return lags_df

def get_X_det(y, forecast_steps = 1):
    """
    This function returns the in-sample and out-of-sample inputs of the deterministic 
    model by instancing time index features for trend and seasonality, if the 
    components have significant variation.
    
    Args:
        y (pd.Series): time series to compute trend and seasonality components.
        forecast_steps (int): number of out-of-sample observations to 
                              generate deterministic inputs.

    Returns:
        X (pd.DataFrame): dataframe the in-sample inputs for the deterministic model.
        X_fore (pd.DataFrame): dataframe the out-of-sample inputs for the 
                               deterministic model.
    """
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
    """
    This function fits the 1st stage deterministic model to y and X. It returns
    the fitted model, the fitted values (in-sample predictions) and, if 
    return_residuals = True, also the fitted residuals.
    
    Args:
        y (pd.Series): time series (output) to fit the model.
        X (pd.DataFrame): inputs to fit the model to the time series.
        model (object): model to be fitted with y and X. 
        return_residuals (bool): whether or not to return the fitted residuals.

    Returns:
        model (object): fitted model.
        y_pred (pd.Series): fitted values series.
        residuals (pd.Series): fitted residuals series.
    """
    model.fit(X, y)
    y_pred = pd.Series(model.predict(X), index = y.index)
    if return_residuals:
        residuals = y - y_pred
        return model, y_pred, residuals
    else:
        return model, y_pred

def get_X_res(residuals, lags):
    """
    This function returns the inputs for the 2nd stage residuals model by 
    computing the lag features and moving average statistics of the 
    residuals.
    
    Args:
        residuals (pd.Series): residuals time series.
        lags (int): number of lagged residuals to include as input.

    Returns:
        residuals_ri (pd.Series): reindexed residuals (observations discarded by lags).
        res_lags_last (pd.DataFrame): last observed residual model inputs.
        res_lags (pd.DataFrame): residual model inputs.
    """
    res_lags = make_lags_features(residuals, lags).dropna()
    res_lags_last = res_lags[-1:]
    res_lags_last.index = res_lags_last.index.shift(1)
    residuals_ri, res_lags = residuals.align(res_lags, join='inner', axis=0)
    return residuals_ri, res_lags_last, res_lags

def get_prediction(models, Xs, res_std):
    """
    This function returns the final prediction by aggregating the point pred of the 
    first and second model. Also includes the forecast datetime and confidence 
    intervals for the prediction.
    
    Args:
        models (tuple): tuple containing the 1st stage and 2nd stage fitted models.
        Xs (tuple): tuple of pd.DataFrames of the inputs of the 1st stage and 2nd stage models.
        res_std (float): fitted residual STD

    Returns:
        pred (dict): dict containing the point forecast and confidence intervals.
    """
    model_1 = models[0]
    model_2 = models[1]
    
    X_1 = Xs[0]
    X_2 = Xs[1]
    
    point_pred = model_1.predict(X_1)[0] + model_2.predict(X_2)[0]
    pred = {
        'point': point_pred,
        'lower': point_pred - 1.96*res_std,
        'upper': point_pred + 1.96*res_std,
        'index': X_1.index[0]
    }
    
    return pred
