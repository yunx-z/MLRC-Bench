class BaseMethod(object):
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def run(self, **args):
        raise NotImplementedError
