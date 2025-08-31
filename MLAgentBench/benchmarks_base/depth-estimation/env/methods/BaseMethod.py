class BaseMethod:
    def __init__(self, name):
        self.name = name
            
    def train(self, mode):
        raise NotImplementedError

    def run(self, mode):
        raise NotImplementedError