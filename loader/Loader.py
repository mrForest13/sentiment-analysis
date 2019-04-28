class Loader(object):

    def __init__(self):
        self.col_names = ['sentiment', 'text']
        self.data = None

    def load(self):
        raise NotImplementedError("Please Implement this method")

    def get_data(self):
        return self.data
