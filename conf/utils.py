
import pandas as pd

def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)

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

def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)