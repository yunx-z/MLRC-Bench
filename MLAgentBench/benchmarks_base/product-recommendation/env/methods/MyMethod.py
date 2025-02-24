import pandas as pd
from collections import defaultdict, Counter

from constants import PREDS_PER_SESSION
from methods.BaseMethod import BaseMethod

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def str2list(self, x):
        x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
        l = [i for i in x.split() if i]
        return l
    
    def run(self):
        df_sess = self.read_train_data()
        df_test = self.read_test_data()

        next_item_dict = defaultdict(list)

        for _, row in df_sess.iterrows():
            prev_items = self.str2list(row['prev_items'])
            next_item = row['next_item']
            prev_items_length = len(prev_items)
            if prev_items_length <= 1:
                next_item_dict[prev_items[0]].append(next_item)
            else:
                for i, item in enumerate(prev_items[:-1]):
                    next_item_dict[item].append(prev_items[i+1])
                next_item_dict[prev_items[-1]].append(next_item)

        for _, row in df_test.iterrows():
            prev_items = self.str2list(row['prev_items'])
            prev_items_length = len(prev_items)
            if prev_items_length <= 1:
                continue
            else:
                for i, item in enumerate(prev_items[:-1]):
                    next_item_dict[item].append(prev_items[i+1])
        
        next_item_map = {}

        for item in next_item_dict:
            counter = Counter(next_item_dict[item])
            next_item_map[item] = [i[0] for i in counter.most_common(100)]

        k = []
        v = []

        for item in next_item_dict:
            k.append(item)
            v.append(next_item_dict[item])
            
        df_next = pd.DataFrame({'item': k, 'next_item': v})
        df_next = df_next.explode('next_item').reset_index(drop=True)

        top200 = df_next['next_item'].value_counts().index.tolist()[:200]

        df_test['last_item'] = df_test['prev_items'].apply(lambda x: self.str2list(x)[-1])
        df_test['next_item_prediction'] = df_test['last_item'].map(next_item_map)

        preds = []

        for _, row in df_test.iterrows():
            pred_orig = row['next_item_prediction']
            pred = pred_orig
            prev_items = self.str2list(row['prev_items'])
            if type(pred) == float:
                pred = top200[:100]
            else:
                if len(pred_orig) < 100:
                    for i in top200:
                        if i not in pred_orig and i not in prev_items:
                            pred.append(i)
                        if len(pred) >= 100:
                            break
                else:
                    pred = pred[:100]
            preds.append(pred)

        df_test['next_item_prediction'] = preds

        return df_test[['locale', 'next_item_prediction']]
