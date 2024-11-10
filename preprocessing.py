# Suppress specific warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Data manipulation and numerical operations
import pandas as pd
import numpy as np

# Scikit-learn utilities for data processing and modeling
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, TimeSeriesSplit

# System and memory management
import os
import gc

# Utility modules
import contextlib
import re
from typing import Optional, Union, Tuple



# Custom Imputer class
class CustomImputer(TransformerMixin):

    """
    A custom imputer class for handling missing values in numerical and categorical columns.
    
    - Numerical columns are imputed using the mean of non-missing values.
    - Categorical columns are imputed using the mode of non-missing values.

    Parameters:
    ----------
    numerical_missing_values : list, default ["-", None, np.nan, "NaN", "nan", "Nan", "Unknown"]
        List of values considered as missing in numerical columns.
        
    categorical_missing_values : list, default ["-", None, np.nan, "NaN", "nan", "Nan", "Unknown"]
        List of values considered as missing in categorical columns.
    """
    

    def __init__(self, 
                 numerical_missing_values = ["-", None, np.nan, "NaN", "nan", "Nan", "Unknown"], 
                 categorical_missing_values = ["-", None, np.nan, "NaN", "nan", "Nan", "Unknown"]):
        
        self.numerical_missing_values = numerical_missing_values
        self.categorical_missing_values = categorical_missing_values
        self.numerical_impute_values = {}
        self.categorical_impute_values = {}

    def fit(self, X, y=None):

        # Numerical columns
        for col in X.select_dtypes(include=[np.number]):
            missing_vals = [val for val in self.numerical_missing_values if val in X[col].values or pd.isnull(val)]
            non_missing = X[col][~X[col].isin(missing_vals)]
            if non_missing.empty:
                # If all non-missing values are in the missing list, impute with 0 or some default value
                self.numerical_impute_values[col] = 0  # Example: Replace with 0
            else:
                self.numerical_impute_values[col] = non_missing.mean()
        
        # Categorical columns
        for col in X.select_dtypes(include=['object']):
            missing_vals = [val for val in self.categorical_missing_values if val in X[col].values or pd.isnull(val)]
            non_missing = X[col][~X[col].isin(missing_vals)]
            if non_missing.empty:
                # If all non-missing values are in the missing list, impute with a default category or ""
                self.categorical_impute_values[col] = ""  # Example: Replace with ""
            else:
                self.categorical_impute_values[col] = non_missing.mode().iloc[0]
        
        return self
    

    def transform(self, X):

        X_transformed = X.copy()
        
        # Numerical columns
        for col in self.numerical_impute_values:
            X_transformed[col].replace(self.numerical_missing_values, np.nan, inplace=True)
            X_transformed[col].fillna(self.numerical_impute_values[col], inplace=True)
        
        # Categorical columns
        for col in self.categorical_impute_values:
            X_transformed[col].replace(self.categorical_missing_values, np.nan, inplace=True)
            X_transformed[col].fillna(self.categorical_impute_values[col], inplace=True)
        
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    



# Custom StratifiedKFold class
class StratifiedKFoldReg:

    """
    Stratified K-Fold cross-validator for regression tasks.
    
    This class stratifies continuous target values by binning them into equal-frequency bins, 
    and ensures that the target distribution is maintained across train/test splits.

    Parameters:
    ----------
    n_splits : int, default = 5
        Number of folds for cross-validation.

    n_bins : int, default = 10
        Number of bins to create for stratification of the target variable.
        
    shuffle : bool, default = True
        Whether to shuffle the data before splitting.
        
    random_state : int, default = 42
        Random seed used when shuffling data.

    """

    def __init__(self, 
                 n_splits = 5, 
                 n_bins = 10, 
                 shuffle = True,
                 random_state = 42):

        self.n_splits = n_splits
        self.n_bins = n_bins
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):

        # Create bins using quantiles for equal-frequency binning
        y_binned = np.digitize(y, bins = np.quantile(y, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        
        if self.shuffle:

            np.random.seed(self.random_state)
            indices = np.random.permutation(len(y))

        else:

            indices = np.arange(len(y))
        
        y_binned = y_binned[indices]
        X = X.iloc[indices]
        
        kf = KFold(n_splits=self.n_splits)
        
        for train_index, test_index in kf.split(X, y_binned):

            yield indices[train_index], indices[test_index]
    




# Main PreprocessingTool functionality
class PreprocessingTool:
    

    def __init__(self, 
                 imputer = CustomImputer(),
                 feature_gen_class = None,
                 val_split = False, val_ratio = 0.2, 
                 val_folds = True, n_folds = 5, 
                 forecasting = False, window_size = 30,
                 encoder = OneHotEncoder(sparse_output = False,
                                         handle_unknown = "ignore"),
                 scaler: Optional[StandardScaler] = MinMaxScaler(),
                 prob_type: Optional[str] = None,
                 seed = 42,
                 ):

        """

        Initialize the instance with the given parameters.

        Parameters:

            imputer (object): An imputer object following the Sklearn API, requiring a .fit_transform() method.
                              WARNING: make sure your imputer implementation can handle a mix of numerical and
                                       categorical features.
                              Defaults to CustomImputer(), which uses "mean" strategy for numerical features 
                              and "most_frequent" for categorical features.
                              Set to None to disable imputing.

            feature_gen_class (class): A feature generator class instance requiring a .generate() method, that processes
                                       train (and val data, if needed). Defaults to None.

            val_split (bool): Determines whether to hold out validation data using Sklearn's train_test_split.
                              Default is False.

            val_ratio (float): The ratio of validation data split, applicable only if val_split is True.
                               Default is 0.2.

            val_folds (bool): Specifies whether to use a cross-validation strategy based on the problem type.
                              Defaults to False. If True, Sklearn StratifiedKFold is used for classification and 
                              custom StratifiedKFoldReg for regression.

            n_folds (int): The number of folds to split the data into, applicable only if val_folds is True.
                           Default is 5.

            forecasting (bool) : Whether task is Time Series Forecasting. If True, it uses special time-based
                                train-val split or Sklearn's TimeSeriesSplit if CV. Input data needs to be 
                                chronologically ordered. Defaults to False.

            window_size (int): The number of time steps to forecast, applicable only if forecasting is True.
                                Default is 30.

            onehot_encoder (object): An encoder object for categorical columns, requiring a .fit_transform() method.
                                     Defaults to sklearn.preprocessing.OneHotEncoder().

            scaler (object): An optional scaler object for numerical columns, requiring a .fit_transform() method.
                             Defaults to sklearn.preprocessing.MinMaxScaler().
                             Set to None to disable scaling.

            prob_type (str): Problem type. Can be: "regression", "binary", "multiclass".
                             If not provided, it is infered automatically from label dtype
                             (experimental).

            seed (int): Random state seed for reproducibility.
                        Default is 42.
        
        """

        self.imputer = imputer

        self.feature_gen_class = feature_gen_class

        self.val_split = val_split
        self.val_ratio = val_ratio
        self.val_folds = val_folds
        self.n_folds = n_folds
        self.forecasting  = forecasting
        self.window_size = window_size

        self.encoder = encoder
        self.label_encoder = None
        self.scaler = scaler

        self.prob_type = prob_type
        self.seed = seed

        self.dropped_columns = None
        self.cat_columns = None
        self.num_columns = None

        self.fitted = False
        self.label = None
    



    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Internal method. Checks input pandas DataFrame df for NaN values.
        If any is found, uses the provided Imputer instance to replace them,
        according to some strategy. Default is CustomImputer with "mean" startegy
        for numerical column feats and "most_frequent" for categorical feats.

        Parameters:

            df (pd.DataFrame): The input pandas DataFrame to be checked.

        Returns:

            (pd.DataFrame): The modified DataFrame.

        """

        print("IMPUTER:\n")

        df = self.imputer.fit_transform(df)

        print("Done. \n-----------------------------------------------------------------------------------------------------------------------------------------")

        return df



    def _get_dtypes(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """
        Internal method. Used to infer features datatypes and make certain conversion(s)
        (like numbers/percentages from string to float). 
        Drops ID column(s) and non-categorical text features.

        Parameters:
            df (pd.DataFrame): Input pandas DataFrame.

        Returns:
            (pd.DataFrame, list): Modified dataframe and remaining features data types list.
        """

        df_dtypes_list = []  # For features data types storing
        conversions_count = 0  # Count for all successful conversions
        dropped_columns = []

        print("DTYPES DETECTOR:\n")
        print("WARNING: current TabularAML implementation doesn't handle non-categorical text features.\n")
        print(f"Found {len(df.dtypes.unique())} unique raw np.dtype(s): {df.dtypes.unique()}.")

        # Conversions and data types storing
        for col in df.columns:
            if df[col].dtype == np.dtype('O'):
                # Check if column is likely categorical based on unique value ratio
                unique_count = df[col].nunique()
                total_count = len(df[col])

                if unique_count / total_count < 0.1:  # Heuristic: Less than 10% unique values -> Likely categorical
                    df[col] = df[col].astype('category')
                    df_dtypes_list.append("cat")
                else:
                    # Try to convert to numeric values (e.g., percentages or numbers in strings)
                    conversion_successful = False
                    for conversion in [
                        lambda x: pd.to_numeric(x),  # Direct numeric conversion
                        lambda x: pd.to_numeric(x.str.replace(',', '')),  # Comma as thousands separator
                        lambda x: pd.to_numeric(x.str.strip().str.rstrip('%')) / 100,  # Percentage
                    ]:
                        try:
                            df[col] = conversion(df[col])
                            conversions_count += 1
                            conversion_successful = True
                            break
                        except (ValueError, TypeError):
                            continue
                    
                    if conversion_successful:
                        df_dtypes_list.append("float" if df[col].dtype == np.float64 else "int")
                    else:
                        # If it's a non-categorical text feature, drop it
                        dropped_columns.append(col)

            elif np.issubdtype(df[col].dtype, np.integer):
                df_dtypes_list.append("int")

            elif np.issubdtype(df[col].dtype, np.floating):
                df_dtypes_list.append("float")

        # Drop ID columns and non-categorical text columns
        for col in df.columns:
            # Drop columns with names like "id" or previously flagged non-categorical text
            if col.lower() == "id" or col in dropped_columns:
                dropped_columns.append(col)
        
        if df.index.name is not None:
            dropped_columns.append(df.index.name)

        # Remove dropped columns from df_dtypes_list
        df_dtypes_list = [dtype for i, dtype in enumerate(df_dtypes_list) if df.columns[i] not in dropped_columns]
        df = df.drop(dropped_columns, axis=1)
        print(f"Dropped {len(dropped_columns)} column(s) with index ID / non-categorical text features.")
        self.dropped_columns = dropped_columns

        # Count remaining features by type
        no_int_feats = sum(1 for dtype in df_dtypes_list if dtype == "int")
        no_float_feats = sum(1 for dtype in df_dtypes_list if dtype == "float")
        no_cat_feats = sum(1 for dtype in df_dtypes_list if dtype == "cat")

        print(f"Converted {conversions_count} column(s) to numeric types.")
        print(f"Remaining features: {no_int_feats} int feat(s), {no_float_feats} float feat(s), {no_cat_feats} categorical feat(s).")
        print("-----------------------------------------------------------------------------------------------------------------------------------------")

        return df, df_dtypes_list






    def _split(self, 
               X: pd.DataFrame,
               y: Union[list, np.ndarray]) -> dict:
    
        
        """
        
        Internal method to split data. Modes: folds / train-test / train only split.

        Parameters:

            X (pd.DataFrame): Input features pandas DataFrame.

            y (Union[list, np.ndarray]): Input labels.


        Returns:

            processed_data (dict): A dictionary containing 2 keys: "data" and "problem type".
                                   The value of key "data" is a list of processed_data. It contains 
                                   n_folds / 1 dictionary element(s) with 'train'/'val' keys and values [X, y].

        """


        # Return dict structure

        processed_data = {
            "data": [],
            "prob_type": self.prob_type
        }


        def process_data_with_feature_generator(X_train, y_train, X_val, y_val, label_column, feature_generator_class):

            """

            Combines features and labels for both train and validation sets, 
            applies feature generation, and then separates features and labels again.
            
            Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series/np.ndarray): Training labels
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series/np.ndarray): Validation labels
            label_column (str): Name of the label column
            feature_generator (class): The feature generation class instance with a .generate() method.
            
            Returns:
            tuple: (X_train_processed, y_train_processed, X_val_processed, y_val_processed)

            """

            # Combine features and labels for train and validation sets
            train_df = X_train.copy()
            train_df[label_column] = y_train
            val_df = X_val.copy()
            val_df[label_column] = y_val

            # Apply feature generation
            train_df_processed, val_df_processed = feature_generator_class.generate(train_df, val_df)

            # Separate features and labels again
            X_train_processed = train_df_processed.drop(label_column, axis=1)
            y_train_processed = train_df_processed[label_column]
            X_val_processed = val_df_processed.drop(label_column, axis=1)
            y_val_processed = val_df_processed[label_column]

            return X_train_processed, y_train_processed, X_val_processed, y_val_processed
            


        # Split df into train-val sets (modes: "folds", "split", "train")

        if self.val_folds:

            if not self.forecasting:
            
                #  Using StratifiedKFold for classification and KFold for regression.       
                if self.prob_type in ["binary", "multiclass"]:

                    print(f"{self.n_folds}-FOLD STRATIFIEDKFOLD TRAIN-VAL SPLITTER:\n")

                    skf = StratifiedKFold(n_splits = self.n_folds, shuffle = True, random_state = self.seed)

                    for train_idx, val_idx in skf.split(X, y):

                        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
                        y_train, y_val = y[train_idx].copy(), y[val_idx].copy()

                        if self.feature_gen_class is not None:
                            X_train, y_train, X_val, y_val = process_data_with_feature_generator(X_train, y_train, 
                                                                                                 X_val, y_val,
                                                                                                 self.label,
                                                                                                 self.feature_gen_class)
                        processed_data["data"].append(
                            {
                                "train": [X_train, y_train],
                                "val": [X_val, y_val]
                            }
                        )


                elif self.prob_type == "regression":

                    print(f"{self.n_folds}-FOLD STRATIFIEDKFOLDREG TRAIN-VAL SPLITTER:\n")
                    
                    skf_reg = StratifiedKFoldReg(n_splits = self.n_folds,
                                                 n_bins = 10, 
                                                 shuffle = True, 
                                                 random_state = self.seed)


                    for train_idx, val_idx in skf_reg.split(X, y):

                        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
                        y_train, y_val = y[train_idx].copy(), y[val_idx].copy()

                        if self.feature_gen_class is not None:
                            X_train, y_train, X_val, y_val = process_data_with_feature_generator(X_train, y_train, 
                                                                                                 X_val, y_val,
                                                                                                 self.label,
                                                                                                 self.feature_gen_class)
                        processed_data["data"].append(
                            {
                                "train": [X_train, y_train],
                                "val": [X_val, y_val]
                            }
                        )

                           
            else:

                print(f"{self.n_folds}-FOLDS TIME SERIES SPLITTER:\n")

                tscv = TimeSeriesSplit(n_splits = self.n_folds, 
                                       test_size = self.window_size)

                # Iterate over each split
                for i, (train_idx, val_idx) in enumerate(tscv.split(X), 1):

                    print(f"Processing fold {i}...")

                    # Split features and labels with copy to avoid modifying original DataFrames
                    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
                    y_train, y_val = y[train_idx].copy(), y[val_idx].copy()

                    if self.feature_gen_class is not None:
                        X_train, y_train, X_val, y_val = process_data_with_feature_generator(X_train, y_train, 
                                                                                            X_val, y_val,
                                                                                            self.label,
                                                                                            self.feature_gen_class)

                    # Append processed data for the current fold
                    processed_data["data"].append(
                        {
                            "train": [X_train, y_train],
                            "val": [X_val, y_val]
                        }
                    )



        elif self.val_split:

            print("TRAIN-VAL SPLITTER:\n")

            if self.forecasting:
                

                no_samples = len(X)
                border = int(round((1-self.val_ratio) * no_samples))
                X_train, X_val = X.iloc[:border], X.iloc[border:]
                y_train, y_val = y[:border], y[border:]
         

            else:

                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = self.val_ratio, random_state = self.seed)

            processed_data["data"].append({
                "train": [X_train, y_train],
                "val": [X_val, y_val]
            })
            
            
        else:

            print("TRAIN-DATA only:\n")

            processed_data["data"].append({
                "train": [X, y]
            })

        print("Done.")
        print("-----------------------------------------------------------------------------------------------------------------------------------------")

        gc.collect()
        return processed_data
                
    

    

    def _encode(self, 
                df_train: pd.DataFrame, 
                df_val: Optional[pd.DataFrame] = None,
                fit = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        
        """
        
        Internal method to encode categorical features using the provided encoder instance.
        Default is OneHotEncoder().

        Parameters:

            df_train (pd.DataFrame): Train pandas DataFrame to .fit_transform() if fit = True.

            df_val (pd.DataFrame): Optional. Validation DataFrame to .transform().

            fit (bool): Whether to fit. Use True for train data, False for inference.
                        Default is True.

        Returns:
            pd.DataFrame: Inplace transformed training DataFrame.

            Optional[pd.DataFrame]: Transformed validation DataFrame, if df_val is provided.

        """

        # Select categorical columns
        categorical_columns = df_train.select_dtypes(include=["object", "category"]).columns
        self.cat_columns = categorical_columns

        # Get indices of categorical columns
        cat_idx = [df_train.columns.get_loc(col) for col in categorical_columns]

        # Check if there are categorical columns to encode
        if len(categorical_columns) > 0:
            # Transform training data
            if fit:
                new_train_cols = self.encoder.fit_transform(df_train.iloc[:, cat_idx]) # for training
            else:
                new_train_cols = self.encoder.transform(df_train.iloc[:, cat_idx]) # for inference

            new_train_cols = pd.DataFrame(new_train_cols, columns=self.encoder.get_feature_names_out(df_train.columns[cat_idx]), index=df_train.index)

            df_train = df_train.drop(columns=df_train.columns[cat_idx])
            df_train = pd.concat([df_train, new_train_cols], axis=1)

            # Clean the column names (for XGB)
            df_train.columns = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_') for col in df_train.columns]

            if df_val is not None:
                # Transform validation data if it exists and has categorical columns
                new_val_cols = self.encoder.transform(df_val.iloc[:, cat_idx])
                new_val_cols = pd.DataFrame(new_val_cols, columns=self.encoder.get_feature_names_out(df_val.columns[cat_idx]), index=df_val.index)

                df_val = df_val.drop(columns=df_val.columns[cat_idx])
                df_val = pd.concat([df_val, new_val_cols], axis=1)

                # Clean the column names (for XGB)
                df_val.columns = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_') for col in df_val.columns]

                return df_train, df_val
            else:
                return df_train
        else:
            # Handle case where there are no categorical columns to encode
            if df_val is not None:
                return df_train, df_val
            else:
                return df_train


    

    def _scale(self,
               df_train: pd.DataFrame,
               df_val: Optional[pd.DataFrame] = None,
               fit = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    

        """
        
        Internal method to scale numerical features using the provided scaler instance.
        Default is MinMaxScaler().

        Parameters:

            df_train (pd.DataFrame): Train pandas DataFrame to .fit_transform() if fit = True.

            df_val (pd.DataFrame): Optional. Validation DataFrame to .transform().

            fit (bool): Whether to fit. Use True for train data, False for inference.
                        Default is True.

        Returns:
            pd.DataFrame: Inplace transformed training DataFrame.

            Optional[pd.DataFrame]: Transformed validation DataFrame, if df_val is provided.

        """

        # Select numerical columns
        num_columns = df_train.select_dtypes(include=np.number).columns
        self.num_columns = num_columns

        # Check if there are numerical columns
        if len(num_columns) > 0:
            # Transform training data
            if fit:
                df_train[num_columns] = self.scaler.fit_transform(df_train[num_columns])
            else:
                df_train[num_columns] = self.scaler.transform(df_train[num_columns])

            if df_val is not None:
                # Transform validation data if it exists and has numerical columns
                df_val[num_columns] = self.scaler.transform(df_val[num_columns])
                
                return df_train, df_val
            else:
                return df_train
        else:
            # Handle case where there are no numerical columns to scale
            if df_val is not None:
                return df_train, df_val
            else:
                return df_train
    


    def fit_transform(self, 
                      df: pd.DataFrame,
                      label: str) -> dict:


        """
        Main method. 

        Infers features data types and problem type if not provided.

        Fits and transforms the input pd.DataFrame.
        The transformations used are those passed through ProcessingTool:
        imputer, splitter, encoder, scaler etc.

        Parameters:

            df (pd.DataFrame): Input pandas DataFrame to be used during 
            train-validation process in Trainer.

            label (str): The column label name.

        Returns:

            processed_data (dict): A dictionary containing 2 keys: "data" and "problem type".
                                   The value of key "data" is a list of processed_data. It contains 
                                   n_folds / 1 element with 'train'/'val' keys and values [X, y].
        
        """

        # Separating feats and labels
        X = df.drop(label, axis=1)
        y = df[label].values
        self.label = label

        # Infering problem type if not provided

        pt_infered = False

        if self.prob_type is None:

            if (np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) / len(y) < 0.1) or y.dtype == np.dtype('O'):

                if len(np.unique(y)) == 2:
                    self.prob_type = "binary"

                if len(np.unique(y)) > 2:
                    self.prob_type = "multiclass"

            else:
                self.prob_type = "regression"

            pt_infered = True
        

        # Use label encoder (needed for TF NN)

        if self.prob_type == "binary" or self.prob_type == "multiclass":

            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        
        print(f"LABEL: {label}\nProblem type: {self.prob_type}.", end=" ")

        if pt_infered:
            print("Was infered from label.\n")

        print("\n-----------------------------------------------------------------------------------------------------------------------------------------")


        
        # Imputing NaN values

        if self.imputer is not None:
            X = self._impute(X)

        # Infering feats dtypes

        X, dtypes_list = self._get_dtypes(X)

        # Splitting data (train-val) according to choosen strategy

        processed_data = self._split(X, y)


        # Scaling numerical feats (based on split mode)

        print(f"SCALER:\n")

        if self.scaler is not None:

            if self.val_folds:

                for i in range(self.n_folds):

                    processed_data["data"][i]["train"][0], processed_data["data"][i]["val"][0] = self._scale(processed_data["data"][i]["train"][0],
                                                                                                             processed_data["data"][i]["val"][0],
                                                                                                             fit = True)
            elif self.val_split:

                processed_data["data"][0]["train"][0], processed_data["data"][0]["val"][0] = self._scale(processed_data["data"][0]["train"][0], 
                                                                                                         processed_data["data"][0]["val"][0],
                                                                                                         fit = True)
            else:

                processed_data["data"][0]["train"][0] = self._scale(processed_data["data"][0]["train"][0], fit = True)

            print("Done.\n-----------------------------------------------------------------------------------------------------------------------------------------")
    
    
    
        # Encoding categorical feats (based on split mode)

        print(f"ENCODER:\n")

        if self.encoder is not None:

            if self.val_folds:

                for i in range(self.n_folds):

                    processed_data["data"][i]["train"][0], processed_data["data"][i]["val"][0] = self._encode(processed_data["data"][i]["train"][0],
                                                                                                              processed_data["data"][i]["val"][0],
                                                                                                              fit = True)
            elif self.val_split:

                processed_data["data"][0]["train"][0], processed_data["data"][0]["val"][0] = self._encode(processed_data["data"][0]["train"][0], 
                                                                                                          processed_data["data"][0]["val"][0],
                                                                                                          fit = True)
            else:

                processed_data["data"][0]["train"][0] = self._encode(processed_data["data"][0]["train"][0], fit = True)

        self.fitted = True
        print("Done.\n-----------------------------------------------------------------------------------------------------------------------------------------")
        
        return processed_data 
    
    


    def transform(self, 
                  df: pd.DataFrame,
                  fit = False) -> dict:

        """
        Transforms the input pd.DataFrame. 
        The transformations used are those passed through ProcessingTool:
        imputer, splitter, encoder, scaler etc. (fitted after using fit_transform())

        Parameters:

            df (pd.DataFrame): Input pandas DataFrame to be used during 
            final evaluation / leaderboard process in Trainer (if it contains label),
            or inference on test data (if not).

        Returns:

            pd.DataFrame: Inplace transformed features DataFrame

            Optional[np.ndarray, list]: Labels.
        
        """

        # No logging 
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):


                # Check if .fit_transform() was called before
                if self.fitted is False:
                    raise Exception("Transformers weren't fitted. Please call .fit_transform() on train data first.")
                
                # Generate new features if needed
                if self.feature_gen_class is not None:
                    df = self.feature_gen_class.generate(df)
                
                # Handle the 2 use cases
                if self.label in df.columns:
                    X = df.drop(self.label, axis=1).copy()
                    y = df[self.label].values
                
                else:
                    X = df.copy()
                    y = None
                        
                # Imputer
                if self.imputer is not None:
                    X = self._impute(X)

                # Drop unwanted columns
                X, dtypes_list = self._get_dtypes(X)
        

                # Scale
                if self.scaler is not None:
                    X = self._scale(df_train = X,
                                    df_val = None,
                                    fit = fit)
                    
                # Encode 
                if self.encoder is not None:
                    X = self._encode(df_train = X,
                                    df_val = None,
                                    fit = fit)
            

                # Encode labels and return modified objects
                if y is not None:
                    
                    if (self.prob_type == "binary" or self.prob_type == "multiclass"):
                        y = self.label_encoder.transform(y)

                    return X, y
                
                else:

                    return X
                