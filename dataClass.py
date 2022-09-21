import pandas as pd

class DataSet:
    def __init__(self):
        self.df = pd.DataFrame()
        self.model = ""
        self.matrix = ""
        self.dict_index = {}
        self.dict_values = {}
