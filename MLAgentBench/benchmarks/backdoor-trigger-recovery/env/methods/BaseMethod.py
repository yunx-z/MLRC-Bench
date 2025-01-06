class BaseMethod(object):
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def run(self, target_list, **args):
        predictions = {}
        for target in target_list:
            pred_list = ["xxxxxxx", "xxxxxxx"] # placeholder for two predicted triggers
            predictions[target] = pred_list
        return predictions

