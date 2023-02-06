
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor

class MyModel():
    def __init__(self):
        self.final_predictions = {}
        self.series_to_id_dict = {}
        self.id_to_market_dict = {}
        self.product_to_series_dict = {}
        self.mape_desc = {}
        self.forecast_date = '2022-12'
    
    def fit(self, prices_dataset):
        """
        This method fits the two stage model to all the time series present in 
        prices_dataset and saves the next-month predictions for inference time.

        Args:
            prices_dataset (pd.Dataframe): dataframe containing the historical
                                           data from all the time series to 
                                           forecast.

        Returns:
            None
        """
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
            X_1, X_fore_1 = self.get_X_det(y, forecast_steps = 1)
            # Get model, fitted values and residuals from deterministic model
            model_1, y_pred_1, residuals = self.get_model(y, X_1, LinearRegression(), return_residuals = True)
            # Save point forecast for deterministic model
            pred = {
                'point_det':model_1.predict(X_fore_1)[0]
            }
            predictions[series_id] = pred

            # Get response variable and features (lags) for residual model
            y_2, X_fore_2, X_2 = self.get_X_res(residuals, lags=3)
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
        self.forecast_date = X_fore_1.index[0].strftime('%Y-%m')

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
        model_2, multi_y_pred_2 = self.get_model(multi_y_2, trans_multi_X_2, LinearRegression(), return_residuals=False)

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
        self.final_predictions = {
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
        self.series_to_id_dict = prices_dataset[['product', 'market', 'series_id']].drop_duplicates().set_index(['product', 'market']).squeeze().to_dict()
        ## series_id -> market
        self.id_to_market_dict = {v:k[1] for k,v in self.series_to_id_dict.items()}
        ## product -> list(series_ids) product in different markets
        self.product_to_series_dict = prices_dataset[['product', 'series_id']].drop_duplicates().groupby(['product'])['series_id'].apply(lambda x: list(np.unique(x))).to_dict()
    
    def evaluate(self, test_dataset):
        """
        This method computes the MAPE error for each time serie that the model 
        was fitted on and computes aggregate statistics. 

        Args:
            test_dataset (pd.DataFrame): pandas dataframe containing the true 
                                         value (last observed price) for each 
                                         time serie that the model was fitted on.

        Returns:
            mape_desc (dict): dictionary containing the aggregate statistics of 
                              the time series MAPE error on the test set.
        """
        test_dataset_copy = test_dataset.copy()
        
        test_dataset_copy['pred_price'] = test_dataset_copy.apply(lambda x: self.predict_product_market(x['product'], x['market']), axis = 1).apply(lambda x: x.get('point'))
        
        test_dataset_copy['mape'] = (abs(test_dataset_copy['mean_price'] - test_dataset_copy['pred_price']) / test_dataset_copy['mean_price']).astype(float)
        
        mape_desc = test_dataset_copy['mape'].describe().apply(lambda x:np.round(x, 3)).to_dict()
        
        self.mape_desc = {'mape_'+key: value for key,value in mape_desc.items() if key != 'count'}

        return mape_desc
    
    def make_lags(self, ts, lags):
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

    def make_lags_features(self, ts, lags):
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

    def get_X_det(self, y, forecast_steps = 1):
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

    def get_model(self, y, X, model, return_residuals):
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

    def get_X_res(self, residuals, lags):
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
        res_lags = self.make_lags_features(residuals, lags).dropna()
        res_lags_last = res_lags[-1:]
        res_lags_last.index = res_lags_last.index.shift(1)
        residuals_ri, res_lags = residuals.align(res_lags, join='inner', axis=0)
        return residuals_ri, res_lags_last, res_lags

    def get_prediction(self, models, Xs, res_std):
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

    def predict_product(self, product):
        """
        This method returns a dict of the forecasted prices for a specified 
        product in all the markets it is traded.

        Args:
            product (string): queried product to get forecasted prices for.

        Returns:
            predictions (dict): dict containing the point forecast and intervals 
                                for every market the specified product trades.
        """
        series_for_product = self.product_to_series_dict.get(product, [])
        predictions = {}

        for series_id in series_for_product:
            market = self.id_to_market_dict[series_id]
            predictions[market] = {
                'point': self.final_predictions[series_id]['point'],
                'lower': self.final_predictions[series_id]['lower'],
                'upper': self.final_predictions[series_id]['upper']
            }

        return predictions

    def predict_product_market(self, product, market):
        """
        This method returns a dict of the forecasted price for a specified 
        product in a specific market.

        Args:
            product (string): queried product to get forecasted prices for.
            market (string): queried market to get forecasted prices for.

        Returns:
            prediction (dict): dict containing the point forecast and intervals 
                               for the specified product and market.
        """
        series_for_product_market = self.series_to_id_dict.get((product, market), -1)

        prediction = {}

        if series_for_product_market > 0:
            prediction['point'] = self.final_predictions[series_for_product_market]['point']
            prediction['lower'] = self.final_predictions[series_for_product_market]['lower']
            prediction['upper'] = self.final_predictions[series_for_product_market]['upper']

        return prediction

    def get_model_info(self):
        """
        This method returns a dict containing information about the fitted model, 
        including the forecast date and last month MAPE error statistics.

        Returns:
            model_info (dict): dict containing model info.
        """
        model_info = {
            'model_name': 'Next-Month Colombian Fruit and Veg Prices Predictor',
            'model_version': '1',
            'forecast_date': self.forecast_date,
            'last_month_mape': self.mape_desc
        }

        return model_info