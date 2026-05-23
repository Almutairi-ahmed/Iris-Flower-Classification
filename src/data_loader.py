import pandas as pd
from sklearn.model_selection import train_test_split


class IrisDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    
    def load_data(self):
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        self.df = pd.read_csv(self.file_path,  names=column_names)
        return self.df
    
    
    def get_features_target(self):
        if self.df is None:
            self.load_data()
        
        x = self.df.drop('species', axis=1) 
        y = self.df['species']
        return x, y
    
    
    def split_data(self, test_size=0.2, random_state=42):
       
        x, y = self.get_features_target()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test