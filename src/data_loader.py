import pandas as pd
from sklearn.model_selection import train_test_split


class IrisDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    
    def load_data(self): #  1 colum       2 columns      3 columns     4 columns     5 columns
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        self.df = pd.read_csv(self.file_path,  names=column_names)
        return self.df
    
    
    def get_features_target(self):
        if self.df is None:
            self.load_data()
        
        x = self.df.drop('species', axis=1) #drop the column 'species' from the dataframe and store the remaining columns in x
        y = self.df['species'] #only the value of the column 'species' will be stored in y
        # x will be what the mode learn from 
        # y will be what the model will predict  
        return x, y
    
    
    
    def split_data(self, test_size=0.2, random_state=42):
        #test_size is the proportion of the dataset to include in the test split
        #random_state is the seed used by the random number generator
        #0.2 means 20% of the data will be used for testing and 80% for training
        x, y = self.get_features_target()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test