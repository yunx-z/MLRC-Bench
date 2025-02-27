from functools import lru_cache
import pandas as pd
import numpy as np
from constants import train_data_path, test_data_path

class BaseMethod(object):
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name
    
    @lru_cache(maxsize=1)
    def read_product_data(self):
        return pd.read_csv('data/products.csv')

    @lru_cache(maxsize=1)
    def read_train_data(self):
        return pd.read_csv(train_data_path)

    @lru_cache(maxsize=3)
    def read_test_data(self):
        return pd.read_csv(test_data_path)

    def check_predictions(self, predictions, test_sessions, check_products=False):
        """
        These tests need to pass as they will also be applied on the evaluator
        """
        test_locale_names = test_sessions['locale'].unique()
        for locale in test_locale_names:
            sess_test = test_sessions.query(f'locale == "{locale}"')
            preds_locale =  predictions[predictions['locale'] == sess_test['locale'].iloc[0]]
            assert sorted(preds_locale.index.values) == sorted(sess_test.index.values), f"Session ids of {locale} doesn't match"

            if check_products:
                # This check is not done on the evaluator
                # but you can run it to verify there is no mixing of products between locales
                # Since the ground truth next item will always belong to the same locale
                # Warning - This can be slow to run
                products = self.read_product_data().query(f'locale == "{locale}"')
                predicted_products = np.unique( np.array(list(preds_locale["next_item_prediction"].values)) )
                assert np.all( np.isin(predicted_products, products['id']) ), f"Invalid products in {locale} predictions"

    def run(self, **args):
        raise NotImplementedError
