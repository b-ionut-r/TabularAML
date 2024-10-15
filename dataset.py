import pandas as pd
from preprocessing import PreprocessingTool
from typing import Optional


class TabularDataset:
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 label: Optional[str] = None, 
                 preprocessor = PreprocessingTool(),
                 type = "train"):

        """

        df (pd.DataFrame): Input data. Contains label column.

        label (str): The string name of df's label column.

        preprocessor (PreprocessingTool): The preprocessor used with given params.
                                          Has to be instance of Class PreprocessingTool.
        
        type (str): Tabular Dataset type. Can be either "train" or "eval" or "infer"

        """

        self.df = df
        self.label = label
        self.preprocessor = preprocessor
        self.type = type
   

    def process(self):

        """
        
        Processes the tabular Dataset using the provided PreprocessingTool.
        
        Returns processed data dictionary.
        
        """
        
        if self.type == "train":
            processed_data = self.preprocessor.fit_transform(self.df, self.label)
            return processed_data
        
        elif self.type == "eval":
            X, y = self.preprocessor.transform(self.df)
            return X, y
        
        elif self.type == "infer":
            X = self.preprocessor.transform(self.df)
            return X