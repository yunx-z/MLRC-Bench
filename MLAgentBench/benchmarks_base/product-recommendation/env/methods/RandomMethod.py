import pandas as pd
import numpy as np

from constants import PREDS_PER_SESSION
from methods.BaseMethod import BaseMethod

class RandomMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def random_predicitons(self, locale, sess_test_locale):
        random_state = np.random.RandomState(42)
        products = self.read_product_data().query(f'locale == "{locale}"')
        predictions = []
        for _ in range(len(sess_test_locale)):
            predictions.append(
                list(products['id'].sample(PREDS_PER_SESSION, replace=True, random_state=random_state))
            ) 
        sess_test_locale['next_item_prediction'] = predictions
        sess_test_locale.drop('prev_items', inplace=True, axis=1)
        return sess_test_locale
    
    def run(self):
        test_sessions = self.read_test_data()
        predictions = []
        test_locale_names = test_sessions['locale'].unique()
        for locale in test_locale_names:
            sess_test_locale = test_sessions.query(f'locale == "{locale}"').copy()
            predictions.append(
                self.random_predicitons(locale, sess_test_locale)
            )
        predictions = pd.concat(predictions).reset_index(drop=True)

        self.check_predictions(predictions, test_sessions)
        return predictions
