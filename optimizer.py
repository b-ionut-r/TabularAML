import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)  # Suppresses log output below CRITICAL level. I use a custom logging callback.
from optuna.exceptions import ExperimentalWarning
from scorer import Scorer

# Supress nasty warnings
import warnings
warnings.filterwarnings('ignore', 
                        category=ExperimentalWarning)
warnings.filterwarnings('ignore', 
                        category=UserWarning, 
                        module='xgboost')
warnings.filterwarnings("ignore", 
                        category=UserWarning,
                        module="lightgbm")
warnings.filterwarnings("ignore", 
                         message="Can't optimze method \"evaluate\" because self argument is used",
                         category=UserWarning)

import lightgbm as lgb
from lightgbm.callback import early_stopping
import xgboost as xgb
import catboost as cat
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.ensemble import VotingClassifier, VotingRegressor
possible_models = [LGBMRegressor, LGBMClassifier,
                   CatBoostRegressor, CatBoostClassifier,
                   XGBRegressor, XGBClassifier,
                   RandomForestRegressor, RandomForestClassifier,
                   SGDRegressor, SGDClassifier,
                   VotingRegressor, VotingClassifier]

import numpy as np
import pandas as pd
import os
import contextlib
import copy


class Optunization:


    def __init__(self, 
                 model: object, 
                 model_hyperparams: dict, 
                 data: list, 
                 eval_metric: Scorer, 
                 early_stopping_rounds: int, 
                 of_mitigation_level: float, 
                 use_gpu: bool, 
                 use_cuda: bool, 
                 n_trials: int, 
                 timeout: int, 
                 seed: int):


        """

        This class creates an OPTUNA study to fine-tune a GBM-model's hyperparams.

        Parameters:

            model (object): A ML Model. Can be: LightGBM, CatBoost, XGBoost, RandomForest or other.
            
            model_hyperparams (dict): The model's hyperparams search space dictionary to be used by Optuna.
                                      Key format: hyperparam name, as expected by the model's constructor.
                                      Value format can be:
                                        - fixed: int, float, string
                                        - hyperparam range tuple:
                                            * (low, high, "int") for integer params;
                                            * (low, high, "float") for floating params;
                                            * (low, high, "float_log") for floating params with logarithmic distribution;
                                            * (categories_list, "cat") for categorical params.
                                      Aditional parameter: model time "priority".
            
            data (list): List of data used for training.
                         Contains n_folds / 1 elements with 'train'/'val' keys and values [X, y].

            eval_metric (object): Instance of class Scorer. Used for early stopping.

            early_stopping_rounds (int): Number of rounds with no positive gain, after which the model stops
                                         early to prevent overfitting.
            
            of_mitigation_level (float): How much to penalize train-val metric gap during Optuna study optimization.
                                         Used to prevent overfitting.

            use_gpu (bool): Whether to use any GPU for training.

            use_cuda (bool): Whether to use Nvidia CUDA GPU training.

            n_trials (int): Number of trials to perform in the Optuna Study.

            timeout (int): Number of seconds after which study stops automatically, even if n_trials isn't reached.

            seed (int): Random state seed to use for training the model. It ensures a level of reproducibility.


        """

        self.model = model
        self.model_hyperparams = model_hyperparams
        self.data = data
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.of_mitigation_level = of_mitigation_level
        self.use_gpu = use_gpu
        self.use_cuda = use_cuda
        self.n_trials = n_trials
        self.timeout = timeout
        self.seed = seed

        # Training mode: folds / train-val split / train-data only
        if len(self.data) > 2:
            self.mode = "folds"
        elif len(self.data[0]) == 2:
            self.mode = "split"
        else:
            self.mode = "train"


        self.study = None
        self.device = None
        self.user_attrs = None
        self.abrv = None
        self.lgb_eval = None
        self.xgb_eval = None
        self.cat_eval = None
        self.rf_eval = None
        self.sgd_lin_eval = None
        self.fit_kwargs = None

        self.xgb_callback = None





    def _objective(self, trial) -> float:

        """
        Optuna Objective function. 

        Returns a metric that Optuna study aims to maximize / minimize.
        
        """

        # Define Optuna param search space

        params = {}

        for name, values in self.model_hyperparams.items():
            if name != "priority":
                if isinstance(values, tuple):
                    if values[-1] == "int":
                        params[name] = trial.suggest_int(name, values[0], values[1])
                    if values[-1] == "float":
                        params[name] = trial.suggest_float(name, values[0], values[1])
                    if values[-1] == "float_log":
                        params[name] = trial.suggest_float(name, values[0], values[1], log=True)
                    if values[-1] == "cat":
                        params[name] = trial.suggest_categorical(name, values[0])
                else:
                    params[name] = trial.suggest_categorical(name, [values]) # fixed value
                    # keep as categorical sampling to ensure its logging
                
        
        # Instantiate GBM model with these and additional model-specific params

        if self.model.__module__ == "lightgbm.sklearn":

            self.abrv = "LGB"

            if self.use_gpu:
                self.device = "gpu"
            else:
                self.device = "cpu"
            
            if self.eval_metric is not None:
                self.lgb_eval = copy.deepcopy(self.eval_metric)
                self.lgb_eval.type = "lgb"
                self.fit_kwargs = {
                    "eval_metric": self.lgb_eval,
                }
                if self.early_stopping_rounds:
                    self.fit_kwargs["callbacks"] = [early_stopping(stopping_rounds = self.early_stopping_rounds, first_metric_only=True)]

                
            self.model.__init__(**params, device_type = self.device, seed = self.seed)


        elif self.model.__module__ == "catboost.core":

            self.abrv = "CAT"

            if self.use_gpu or self.use_cuda:
                self.device = "GPU"
            else:
                self.device = "CPU"
    
            eval_vars = vars(self.eval_metric)
            eval_vars["type"] = "cat"
            self.cat_eval = type(self.eval_metric)(**eval_vars)
            self.fit_kwargs = {
                "metric_period": 5
            }
            if self.early_stopping_rounds:
                self.fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

    
            self.model.__init__(**params, task_type = self.device, random_seed = self.seed, eval_metric=self.cat_eval)


        elif self.model.__module__ == "xgboost.sklearn":

            self.abrv = "XGB"

            if self.use_cuda:
                self.device = "cuda"

            if self.eval_metric is not None:
                self.xgb_eval = copy.deepcopy(self.eval_metric)
                self.xgb_eval.type = "xgb"
                self.xgb_callback = (xgb.callback.EarlyStopping(rounds = self.early_stopping_rounds,
                                                                data_name = 'validation_0',
                                                                metric_name = self.eval_metric.name,
                                                                maximize = self.eval_metric.greater_is_better,
                                                                save_best = False))
                self.fit_kwargs = {
                    "eval_metric": self.xgb_eval,
                }
                if self.early_stopping_rounds:
                    self.fit_kwargs["callbacks"] = [self.xgb_callback]
            
            self.model.__init__(**params, device = self.device, seed = self.seed)


        elif self.model.__module__ == "sklearn.ensemble._forest":

            self.abrv = "RF"


            if self.eval_metric is not None:
                self.rf_eval = copy.deepcopy(self.eval_metric)

            self.model.__init__(**params, random_state = self.seed)
            

        elif self.model.__module__ == "sklearn.linear_model._stochastic_gradient":

            self.abrv = "SGD_LINEAR"


            if self.eval_metric is not None:
                self.sgd_lin_eval = copy.deepcopy(self.eval_metric)

            self.model.__init__(**params, random_state = self.seed)

        # Supress all fit verbosity, lgbm warnings etc.
        with open(os.devnull, 'w') as devnull:
             with contextlib.redirect_stdout(devnull):

        # with do_nothing():
        #      with do_nothing():

                score = 0
                metr_name = self.eval_metric.name
                effective_n_estimators_list = None


                ### Folds mode

                if self.mode == "folds": 

                    effective_n_estimators_list = []
                    train_metrics = []
                    val_metrics = []
                    
                    # Train folds

                    for fold in self.data:

                        # Special case to handle repeated XGBoost EarlyStopping

                        if self.abrv == "XGB":

                            self.model.__init__(**params, device = self.device, seed = self.seed)
                            callbacks = None
                            if self.early_stopping_rounds:
                                self.xgb_callback = xgb.callback.EarlyStopping(rounds = self.early_stopping_rounds,
                                                                               data_name = 'validation_0',
                                                                               metric_name = self.eval_metric.name,
                                                                               maximize = self.eval_metric.greater_is_better,
                                                                               save_best = False)
                                callbacks = [self.xgb_callback]
    
                            
                            self.model.fit(fold["train"][0], 
                                           fold["train"][1],
                                           eval_set = [tuple(fold["val"])],
                                           eval_metric = self.xgb_eval,
                                           callbacks = callbacks,
                                          )

                        elif self.abrv in ["RF", "SGD_LINEAR"]:
                            self.model.fit(fold["train"][0],
                                           fold["train"][1])

                        else:

                            self.model.fit(fold["train"][0], 
                                           fold["train"][1],
                                           eval_set = [tuple(fold["val"])],
                                           **self.fit_kwargs)
                            
                

                        # Compute and store train-val eval metrics
                        
                        train_fold_metric = self.eval_metric.score(y_true = fold["train"][1],
                                                                   y_pred = self.model.predict(fold["train"][0]).squeeze())                                    
                        train_metrics.append(train_fold_metric)

                        val_fold_metric = self.eval_metric.score(y_true = fold["val"][1], 
                                                                 y_pred = self.model.predict(fold["val"][0]).squeeze())
                        val_metrics.append(val_fold_metric)


                        # Store effective n_estimators_ per fold
                        if self.early_stopping_rounds:
                            if self.abrv == "LGB":
                                effective_n_estimators_list.append(self.model.n_estimators_)
                            elif self.abrv == "XGB":
                                effective_n_estimators_list.append(np.array(self.xgb_callback.stopping_history["validation_0"]
                                                                            [self.xgb_callback.metric_name]).argmax(-1) 
                                                                   if self.eval_metric.greater_is_better 
                                                                   else np.array(self.xgb_callback.stopping_history["validation_0"][self.xgb_callback.metric_name]).argmin(-1))
                            elif self.abrv == "CAT":
                                effective_n_estimators_list.append(self.model.get_best_iteration())


                    # Compute aggregated metric (with overfitting mitigation)
                    # display(f"Train metrics: {train_metrics}.   Val metrics: {val_metrics}.")
                    mean_train_metric = np.mean(np.array(train_metrics), axis=0)
                    mean_val_metric = np.mean(np.array(val_metrics), axis=0)
                    diff = abs(mean_train_metric - mean_val_metric)
                    score = (mean_val_metric - self.of_mitigation_level * diff if self.eval_metric.greater_is_better
                            else mean_val_metric + self.of_mitigation_level * diff)



                    # Save trial's custom user attrs (mean metrics per fold, optimized metric)

                    if self.of_mitigation_level:

                        trial.set_user_attr("mean_train_" + metr_name, mean_train_metric)
                        trial.set_user_attr("mean_val_" + metr_name, mean_val_metric)
                        
                        trial.set_user_attr("optimized_metric", score)

                    else:

                        trial.set_user_attr("mean_train_" + metr_name, mean_train_metric)
                        trial.set_user_attr("mean_val_" + metr_name, mean_val_metric)


                    
                
                ### Split mode

                if self.mode == "split":

                    # Train

                    if self.abrv in ["RF", "SGD_LINEAR"]:
                        self.model.fit(self.data[0]["train"][0],
                                       self.data[0]["train"][1])
                    else:
                        self.model.fit(self.data[0]["train"][0],
                                       self.data[0]["train"][1],
                                       eval_set = [tuple(self.data[0]["val"])],
                                       **self.fit_kwargs)


                    # Compute train-val metrics and optimized metric (with overfitting mitigation)

                    train_metric = self.eval_metric.score(y_true = self.data[0]["train"][1],
                                                          y_pred = self.model.predict(self.data[0]["train"][0]).squeeze())
                    
                    val_metric = self.eval_metric.score(y_true = self.data[0]["val"][1], 
                                                        y_pred = self.model.predict(self.data[0]["val"][0]).squeeze())
                    
                    diff = abs(train_metric - val_metric)
                    score = (val_metric - self.of_mitigation_level * diff if self.eval_metric.greater_is_better
                            else val_metric + self.of_mitigation_level * diff)



                    # Save trial's custom user attrs (train / val metrics, optimized metric)

                    if self.of_mitigation_level:
                        trial.set_user_attr("train_" + metr_name, train_metric)
                        trial.set_user_attr("val_" + metr_name, val_metric)
                        trial.set_user_attr("optimized_metric", score)
                    else:
                        trial.set_user_attr("train_" + metr_name, train_metric)
                        trial.set_user_attr("val_" + metr_name, val_metric)




                ### Train-data only mode, no validation
                    

                if self.mode == "train":

                    # Train

                    if self.abrv in ["RF", "SGD_LINEAR"]:
                        self.model.fit(self.data[0]["train"][0], self.data[0]["train"])
                    else:
                        self.model.fit(self.data[0]["train"][0], self.data[0]["train"],
                                       eval_set = [tuple(self.data[0]["train"])],
                                       **self.fit_kwargs)

                    # Compute train metric

                    train_metric = self.eval_metric.score(y_true = self.data[0]["train"][1], 
                                                          y_pred = self.model.predict(self.data[0]["train"][0]).squeeze())
                    
                    # Use train metric for optimization
                    score = train_metric
                    trial.set_user_attr("train_" + metr_name, train_metric)





                # Store effecting number of estimators / boosting rounds as custom trial attr
                # Use mean and round if mode is "folds"

                
                if self.early_stopping_rounds:

                    if self.abrv == "LGB":

                        effective_n_estimators = (int(np.mean(np.array(effective_n_estimators_list)).round()) 
                                                if effective_n_estimators_list is not None 
                                                else self.model.n_estimators_)
                        
                    elif self.abrv == "XGB":

                        effective_n_estimators = (int(np.mean(np.array(effective_n_estimators_list)).round()) 
                                if effective_n_estimators_list is not None 
                                else np.array(self.xgb_callback.stopping_history["validation_0"][self.xgb_callback.metric_name]).argmax(-1)
                                    if self.eval_metric.greater_is_better 
                                    else np.array(self.xgb_callback.stopping_history["validation_0"][self.xgb_callback.metric_name]).argmin(-1))

                        
                    elif self.abrv == "CAT":


                        effective_n_estimators = (int(np.mean(np.array(effective_n_estimators_list)).round()) 
                                                if effective_n_estimators_list is not None 
                                                else self.model.get_best_iteration())

                    elif self.abrv == "RF":

                        effective_n_estimators = self.model.n_estimators
                    
                    elif self.abrv == "SGD_LINEAR":
                        effective_n_estimators = None

                    trial.set_user_attr("params_n_estimators_", effective_n_estimators)


        return score  
    

    


    def log_trial_callback(self, _, trial): 

        """

        This callback logs information at each Optuna Trial end.

        """ 

        values = dict(trial.user_attrs)
        params = dict(trial.params)
        if self.early_stopping_rounds:
            params["n_estimators_"] = values.pop("params_n_estimators_")

        print(f'Trial {trial.number} finished with values: {values},\nand parameters: {params}.\n')
        self.user_attrs = list(trial.user_attrs.keys())
        if self.early_stopping_rounds:
            self.user_attrs.remove("params_n_estimators_")



  
    def _get_trial_params(self) -> list:

        """

        Retrieves model constructor hyperparams from Optuna trials.

        Returns a list of dictionaries.

        """

        params_list = []

        for trial in self.study.trials:

            trial_params = trial.params

            # Replace n_estimators with effective n_estimators_ after early_stopping
            if self.early_stopping_rounds:
                trial_params["n_estimators"] = trial.user_attrs["params_n_estimators_"] 


            # Model-specific params

            if self.model.__module__ == "lightgbm.sklearn":
                trial_params["device_type"] = self.device
                trial_params["seed"] = self.seed
                
            elif self.model.__module__ == "catboost.core":
                trial_params["task_type"] = self.device
                trial_params["random_seed"] = self.seed

            elif self.model.__module__ == "xgboost.sklearn":
                trial_params["device"] = self.device
                trial_params["seed"] = self.seed

            elif self.model.__module__ == "sklearn.ensemble._forest":
                trial_params["random_state"] = self.seed

            params_list.append(trial_params)

            
        return params_list





    def _prep_model_lb(self) -> pd.DataFrame:

        """

        Returns a leaderboard (a pd.DataFrame with best models found and their hyperparams).

        Modifies Study's trials_dataframe() df to keep only relevant info and integrate custom user attrs.

        Adds a new column with instances of the models.

        """

        df = self.lb_df.copy()

        # Identify columns that start with 'user_attrs'
        user_attr_columns = [col for col in df.columns if col.startswith("user_attrs_")]

        # Rename those columns to remove the prefix
        new_user_attr_columns = [col.replace("user_attrs_", "") for col in user_attr_columns]
        df.rename(columns=dict(zip(user_attr_columns, new_user_attr_columns)), inplace=True)
        if self.early_stopping_rounds:
            new_user_attr_columns.remove("params_n_estimators_")

        # Collect params and other columns
        params_columns = [col for col in df.columns if col.startswith("params_")]
        other_columns = [col for col in df.columns if col not in new_user_attr_columns and col not in params_columns]
        other_columns.remove("number")

        # Reorder columns
        new_column_order = ["id"] + new_user_attr_columns + params_columns + other_columns
        df["id"] = [f"{self.abrv}_" + str(no) for no in df["number"].values]
        df = df[new_column_order] 

        # Add a column with models instances from found hyperparams
        df["model"] = self._get_trial_params()
        # df["model"] = df["model"].apply(lambda param_dict: type(self.model)(**param_dict))
        df["model"] = df["model"].apply(
                        lambda param_dict: type(self.model)(**{k: v for k, v in param_dict.items() if v is not None})
                        )

        # Sort df
        df = df.sort_values("value", ascending = not(self.eval_metric.greater_is_better), ignore_index=True)

        # Drop auto-logged value
        df = df.drop(columns=["value"])

        return df




    def optimize(self) -> pd.DataFrame:

        """
        
        Main method of class Optunization.

        Creates an Optuna study and starts the optimization process.

        Returns a leaderboard df.

        """

        self.study = optuna.create_study(study_name = self.model.__module__,
                                         sampler = optuna.samplers.TPESampler(multivariate=True),
                                         pruner = optuna.pruners.MedianPruner(),
                                         direction = "maximize" if self.eval_metric.greater_is_better else "minimize")
        
        self.study.optimize(self._objective, 
                            n_trials = self.n_trials,
                            timeout = self.timeout, 
                            gc_after_trial=True, 
                            show_progress_bar=True,
                            callbacks = [self.log_trial_callback])
        
        self.lb_df = self.study.trials_dataframe()
        self.lb_df = self._prep_model_lb()

        return self.lb_df
