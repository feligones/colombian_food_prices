
import pandas as pd
import numpy as pd
import pickle
import json

class MyModel():
    def __init__(self, final_predictions, routing_dicts, forecast_date):
        self.final_predictions = final_predictions
        self.series_to_id_dict = routing_dicts['series_to_id_dict']
        self.id_to_market_dict = routing_dicts['id_to_market_dict']
        self.product_to_series_dict = routing_dicts['product_to_series_dict']
        self.forecast_date = forecast_date


    def predict_product(self, product):

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
        series_for_product_market = self.series_to_id_dict.get((product, market), -1)

        prediction = {}

        if series_for_product_market > 0:
            prediction['point'] = self.final_predictions[series_for_product_market]['point']
            prediction['lower'] = self.final_predictions[series_for_product_market]['lower']
            prediction['upper'] = self.final_predictions[series_for_product_market]['upper']

        return prediction

    def get_model_info(self):
        
        model_info = {
            'model_name': 'Next-Month Colombian Fruit and Veg Prices Predictor',
            'model_version': '1',
            'forecast_date': self.forecast_date
        }

        return model_info