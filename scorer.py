from typing import Optional, Dict, Union, List
import numpy as np
import pandas as pd

class CatScorer:

    def __init__(self,
                 name: str,
                 scorer: callable,
                 greater_is_better: bool,
                 extra_params: Dict[str, any], 
                 type: Optional[str] = None):

        """

        Custom eval metric class especially designed for CatBoost API.

        Parameters:

            name (str): Scorer name.

            scorer (obj): Sklearn loss / scoring function from sklearn.metrics or 
                          similar callable object.

            greater_is_better (callable): Whether higher values mean better performance.
                                      Usually, True for scorers used in binary / multiclass tasks 
                                      and False for losses used in regression.

            extra_params (dict): A dictionary of extra kwargs to be passed to Sklearn scorer,
                                 along with y_true and y_pred.
            
            type (str): Model type to be used with. (CAT)
            
        
        """

        self.name = name
        self.scorer = scorer
        self.greater_is_better = greater_is_better
        self.type = type
        self.extra_params = extra_params


    def score(self, 
              y_true: Union[np.ndarray, list],
              y_pred: Union[np.ndarray, list]) -> float:

        return self.scorer(y_true, y_pred, **self.extra_params)
    

    def is_max_optimal(self) -> bool:

        return self.greater_is_better
    

    def evaluate(self, approxes: list, target: list, _) -> tuple[float, int]:

        # Handle CatBoost's specific approxes format
        if len(approxes)==1: # for regression
            y_pred = approxes[0]
            if ((target==0) | (target==1)).all(): # for binary
                y_pred = np.round(y_pred)
        else:
            y_pred = np.vstack(approxes).T.argmax(-1) # for classification

        score = self.score(y_true = target, y_pred = y_pred)

        return score, int(self.greater_is_better)
    

    def get_final_error(self, error: float, _):

        return error






class Scorer:

    def __new__(cls, *args, **kwargs):

        # If model is CatBoost, instantiate custom-signature implementation (CatScorer instance)
        if kwargs.get("type") == "cat":
            return CatScorer(*args, **kwargs)
        

        # Else, for LightGBM and XGBoost, use regular Scorer
        else:
            instance = super().__new__(cls)
            return instance
        

    def __init__(self, 
                 name: str, 
                 scorer: callable, 
                 greater_is_better: bool, 
                 extra_params: Dict[str, any],
                 type: Optional[str] = None):

        """

        Custom eval metric class designed to work with all GBM models.
        Interoperable between LightGBM, XGBoost and Catboost.

        Parameters:

            name (str): Scorer name.

            scorer (callable): Sklearn loss / scoring function from sklearn.metrics.

            greater_is_better (bool): Whether higher values mean better performance.
                                      Usually, True for scorers used in binary / multiclass tasks 
                                      and False for losses used in regression.
            
            extra_params (dict): A dictionary of extra kwargs to be passed to Sklearn scorer,
                                 along with y_true and y_pred.

            type (str): Model type to be used with. (CAT)
        
        """


        self.name = name
        self.scorer = scorer
        self.greater_is_better = greater_is_better
        self.extra_params = extra_params
        self.type = type




    def score(self, 
              y_true: Union[np.ndarray, list],
              y_pred: Union[np.ndarray, list]) -> float:
        
        if len(y_pred.shape) == 2: # check for multiclass
            y_pred = y_pred.argmax(axis=-1)
            
        elif ((y_true==0) | (y_true==1)).all(): # check for binary
            y_pred = np.round(y_pred)
            

        return self.scorer(y_true, y_pred, **self.extra_params)
        
    
    def __call__(self, y1, y2):

        # Handle model-specific return signature.

        if self.type == "lgb":

            # order y_true, y_pred
            y_true = y1
            y_pred = y2
            
            score = self.score(y_true, y_pred)
            return self.name, score, self.greater_is_better
        

        elif self.type == "xgb":
            
            # order y_pred, y_true DMatrix
            y_pred = y1
            y_true = y2.get_label() # from DMatrix
            

            score = self.score(y_true, y_pred)

            return self.name, score
