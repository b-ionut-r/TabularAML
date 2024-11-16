# Import necessary modules

from preprocessing import PreprocessingTool
from dataset import TabularDataset
from scorer import Scorer, predefined_scorers
from optimizer import Optunization
from hyperparams_configs import DEFAULT_CONFIG

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import Bunch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

import optuna
import lightgbm as lgb
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
                   VotingRegressor, VotingClassifier,
                   ElasticNet, ElasticNetCV, LogisticRegression, Ridge]


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', sklearn.exceptions.ConvergenceWarning)
import pandas as pd
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import numpy as np


import os
import dill
import re
import gc
from typing import Union, Dict, Optional, Tuple
from contextlib import contextmanager
import contextlib
@contextmanager
def do_nothing():
    yield


        
# Modify _predict method for Voting ensembles (added .squeeze() for CAT model)
def custom_predict(self, X):
    """Custom implementation to collect results from clf.predict calls."""
    return np.asarray([est.predict(X).squeeze() for est in self.estimators_]).T




class Trainer:

    def __init__(self, 
                 dataset: TabularDataset,
                 eval_dataset: TabularDataset,
                 eval_metric: Optional[Scorer] = None,
                 early_stopping_rounds = 0, # off
                 of_mitigation_level = 0.2,
                 models = ["LGB", "XGB", "CAT", "RF", "SGD_LINEAR"], # ["SKL", "ADA"]
                 hyperparams = DEFAULT_CONFIG,
                 use_gpu = False,
                 use_cuda = True,
                 n_trials = 10000, # max
                 timeout = 3000,
                 seed = 42,
                 select_top = 3,
                 train_meta = True,
                 meta_timeout = 600,
                 save_path = "saved_trainer.pkl"):
        
        
        """

            TabularAML Trainer. Designed to optimize, find, train and ensemble best model
        for your tabular data. Used in both regression and binary / multiclass tasks.

            Parameters:

                dataset (TabularDataset): A TabularAML TabularDataset instance with your data.
                Used in CV hyperparam-tuning (with early stopping) and for refitting models at end on
                the entire data.

                eval_dataset (TabularDataset): A TabularDataset used exclusively for evaluation
                purposes: generating final leaderboard (including Ensemble model(s)).

                eval_metric (Scorer | str): A Scorer instance with the eval metric chosen to be used 
                                            during training for optimization (early stopping) and for evaluation, logging and
                                            ranking purposes.
                                            Can be an abbreviation of a common loss / score, such as "rmse", "rmsle",
                                            "mae", "mse", "r2". 
                                            If no eval_metric is provided, "rmse" will be used by default.  


                early_stopping_rounds (int): Stops training after specified rounds without validation improvement. 
                                             Used for faster training, allowing a deeper exploration of the search space,
                                             possibly at the expense of full reproducibility of validation performance
                                             on the test set.
                                             WARNING: Using this parameter, especially with low non-zero values,
                                                      may cause underfitting and poor reproducibility.
                                                      Consider using `of_mitigation_level` instead for more stable performance.
                                                      Default: 0 (disabled).

                of_mitigation_level (float): Recommended way to prevent overfitting, while also avoiding underfitting.
                                             Penalizes train-validation metric gap during Optuna optimization.
                                             Helps control overfitting without affecting model depth.
                                             Use with disabled early stopping for more consistent results.
                                             Default: 0.2.

                models (list): List of models to use. Select from:
                                    * "LGB" for LGBMRegressor / LGBMClassifier
                                    * "CAT" for CatBoostRegressor / CatBoostClassifier
                                    * "XGB" for XGBRegressor / XGBClassifer
                                    * "RF" for RandomForestRegressor / RandomForestClassifier
                                    * "SGD_LINEAR" for SGDRegressor / SGDClassifier
                                Default is ["LGB", "XGB", "CAT", "RF", "SGD_LINEAR"].

                hyperparams (dict): A dictionary of dictionaries with default hyperparameter search space for each model.
                                    Each model dictionary has the following structure:
                                    Key format: hyperparam name, as expected by the model's constructor.
                                    Value format can be:
                                        - fixed: int, float, string
                                        - hyperparam range tuple:
                                            * (low, high, "int") for integer params;
                                            * (low, high, "float") for floating params;
                                            * (low, high, "float_log") for floating params with logarithmic distribution;
                                            * (categories_list, "cat") for categorical params. 
                                    Additional param: model's time "priority"
                
                use_gpu (bool): Whether to use any GPU for training.
                                Default is True.

                use_cuda (bool): Whether to use Nvidia CUDA GPU training.
                                 Defaults to True.

                n_trials (int): Number of trials to perform in the Optuna Study.
                                Default is 1000 (as high as possible).

                timeout (int): Number of seconds after which all GBMs models tuning stops automatically.
                               Default is 3000s or 50 mins.
                               Default is 3000s or 50 mins.

                seed (int): Random state seed to use for training the model. It ensures a level of reproducibility.
                            Default is 42.

                select_top (int): Select best x models for each model type, as determined by Optuna.
                                  Default is 3.

                train_meta (bool): WWhether to train meta-model on top of the best models of each kind.
                                   Default is True.

                meta_timeout (int): Number of seconds after which meta model study stops automatically.
                                    Default is 600s or 10 mins.

                save_path (str): Path to save the trainer instance to. File extension needs to be .pkl.
                                 Default is "saved_trainer.pkl".
                


        """
        
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.of_mitigation_level = of_mitigation_level
        self.models = models
        self.hyperparams = hyperparams
        self.use_gpu = use_gpu
        self.use_cuda = use_cuda
        self.n_trials = n_trials
        self.timeout = timeout
        self.seed = seed
        self.select_top = select_top
        self.train_meta = train_meta
        self.meta_timeout = meta_timeout
        self.save_path = save_path

        self.processed_data = None
        self.mode = None
        self.pb_type = None


        self.user_attrs = None

        self.lgb_lb = None
        self.cat_lb = None
        self.xgb_lb = None
        self.rf_lb = None
        self.sgd_lin_lb = None
        self.leaderboard = None
        self.eval_lb = None
        self.ensemble = None
        self.best_model = None
        self.X_train_full = None
        self.y_train_full = None
        self.X_eval = None
        self.y_eval = None
        
        self.meta_study = None
        self.meta_dict = None
        self.meta_learner = None

        
        
    
    def train(self):

        """

        Main train method. Starts the optimization process with Optuna, finds best models,
        prints leaderboards, refits top best and ensembles them, preparing them for prediction.
    
        """

        # Process TabularDataset using PreprocessingTool
        self.processed_data = self.dataset.process()
        self.pb_type = self.dataset.preprocessor.prob_type

        # Eval Metric
        if self.eval_metric in predefined_scorers.keys():
            self.eval_metric = predefined_scorers[self.eval_metric]
        if self.eval_metric is None and self.pb_type == "regression":
            self.eval_metric = predefined_scorers["rmse"]


        print(f"\n\n\nSTARTING TRAINING ... \n\n")

        # Computing Optuna timeouts based on models priorities

        total_priority = 0 
        for model, params in self.hyperparams.items():
            if model in self.models:
                total_priority += params["priority"]
        for model, params in self.hyperparams.items():
            if model in self.models:
                params["priority"] = params["priority"] / total_priority




        ### Optuning models

        if "LGB" in self.models:

            print("OPTUNING LGB MODEL...")

            # Task specific additional params

            if self.pb_type == "regression":
                lgb_model = lgb.LGBMRegressor()
            elif self.pb_type in ["binary", "multiclass"]:
                if self.pb_type == "multiclass":
                    self.hyperparams["LGB"]["class_weight"] = (["balanced"], "cat")
                lgb_model = lgb.LGBMClassifier()

            

            # Find best hyperparams with Optuna and display top models

            self.lgb_optuner = Optunization(model = lgb_model, 
                                            model_hyperparams = self.hyperparams["LGB"], 
                                            data = self.processed_data["data"], 
                                            eval_metric = self.eval_metric,
                                            early_stopping_rounds=self.early_stopping_rounds,
                                            of_mitigation_level = self.of_mitigation_level,
                                            use_gpu = self.use_gpu, 
                                            use_cuda = self.use_cuda,
                                            n_trials = self.n_trials,
                                            timeout = self.hyperparams["LGB"]["priority"] * self.timeout,
                                            seed = self.seed
                                            )
            
            self.lgb_lb = self.lgb_optuner.optimize()
            self.mode = self.lgb_optuner.mode
            self.user_attrs = self.lgb_optuner.user_attrs  

            print(f"\n\nFinished training. TOP {self.select_top} models are:")
            display(self.lgb_lb.head(self.select_top))
            print(384 * "-", "\n\n")



        if "XGB" in self.models:

            print("OPTUNING XGB MODEL...\n")


            # Task specific additional params

            if self.pb_type == "regression":
                xgb_model = xgb.XGBRegressor()
            elif self.pb_type in ["binary", "multiclass"]:
                xgb_model = xgb.XGBClassifier()



            # Find best hyperparams with Optuna and display top models

            self.xgb_optuner = Optunization(model = xgb_model, 
                                            model_hyperparams = self.hyperparams["XGB"], 
                                            data = self.processed_data["data"], 
                                            eval_metric = self.eval_metric,
                                            early_stopping_rounds=self.early_stopping_rounds,
                                            of_mitigation_level = self.of_mitigation_level,
                                            use_gpu = self.use_gpu, 
                                            use_cuda = self.use_cuda,
                                            n_trials = self.n_trials,
                                            timeout = self.hyperparams["XGB"]["priority"] * self.timeout,
                                            seed = self.seed
                                            )
            
            self.xgb_lb = self.xgb_optuner.optimize()
            self.mode = self.xgb_optuner.mode
            self.user_attrs = self.xgb_optuner.user_attrs

            print(f"\n\nFinished training. TOP {self.select_top} models are:")
            display(self.xgb_lb.head(self.select_top))
            print(384 * "-", "\n\n")




        if "CAT" in self.models:

            print("OPTUNING CAT MODEL...")

            # Task specific additional params

            if self.pb_type == "regression":
                cat_model = cat.CatBoostRegressor()
            elif self.pb_type in ["binary", "multiclass"]:
                self.hyperparams["CAT"]["auto_class_weights"] = (["Balanced"], "cat")
                cat_model = cat.CatBoostClassifier()



            # Find best hyperparams with Optuna and display top models

            self.cat_optuner = Optunization(model = cat_model, 
                                            model_hyperparams = self.hyperparams["CAT"], 
                                            data = self.processed_data["data"], 
                                            eval_metric = self.eval_metric,
                                            early_stopping_rounds=self.early_stopping_rounds,
                                            of_mitigation_level = self.of_mitigation_level,
                                            use_gpu = self.use_gpu, 
                                            use_cuda = self.use_cuda,
                                            n_trials = self.n_trials,
                                            timeout = self.hyperparams["CAT"]["priority"] * self.timeout,
                                            seed = self.seed
                                            )
            
            self.cat_lb = self.cat_optuner.optimize()
            self.mode = self.cat_optuner.mode
            self.user_attrs = self.cat_optuner.user_attrs

            print(f"\n\nFinished training. TOP {self.select_top} models are:")
            display(self.cat_lb.head(self.select_top))
            print(384 * "-", "\n\n")


        if "RF" in self.models:

            print("OPTUNING RF MODEL...")

            # Task specific additional params

            if self.pb_type == "regression":
                rf_model = RandomForestRegressor()
            else:
                rf_model = RandomForestClassifier()


            # Find best hyperparams with Optuna and display top models

            self.rf_optuner = Optunization(model=rf_model,
                                           model_hyperparams=self.hyperparams["RF"],
                                           data=self.processed_data["data"],
                                           eval_metric=self.eval_metric,
                                           early_stopping_rounds=self.early_stopping_rounds,
                                           of_mitigation_level=self.of_mitigation_level,
                                           use_gpu=self.use_gpu,
                                           use_cuda=self.use_cuda,
                                           n_trials=self.n_trials,
                                           timeout=self.hyperparams["RF"]["priority"] * self.timeout,
                                           seed=self.seed
                                          )

            self.rf_lb = self.rf_optuner.optimize()
            self.mode = self.rf_optuner.mode
            self.user_attrs = self.rf_optuner.user_attrs

            print(f"\n\nFinished training. TOP {self.select_top} models are:")
            display(self.rf_lb.head(self.select_top))
            print(384 * "-", "\n\n")



        if "SGD_LINEAR" in self.models:

            print("OPTUNING SGD LINEAR MODEL...")

            # Task specific additional params

            if self.pb_type == "regression":
                sgd_lin_model = SGDRegressor()
            else:
                self.hyperparams["SGD_LINEAR"]["loss"] = (["log_loss", "modified_huber"], "cat")
                sgd_lin_model = SGDClassifier()

            # Find best hyperparams with Optuna and display top models

            self.sgd_lin_optuner = Optunization(model=sgd_lin_model,
                                                model_hyperparams=self.hyperparams["SGD_LINEAR"],
                                                data=self.processed_data["data"],
                                                eval_metric=self.eval_metric,
                                                early_stopping_rounds=self.early_stopping_rounds,
                                                of_mitigation_level=self.of_mitigation_level,
                                                use_gpu=self.use_gpu,
                                                use_cuda=self.use_cuda,
                                                n_trials=self.n_trials,
                                                timeout=self.hyperparams["SGD_LINEAR"]["priority"] * self.timeout,
                                                seed=self.seed
                                                )

            self.sgd_lin_lb = self.sgd_lin_optuner.optimize()
            self.mode = self.sgd_lin_optuner.mode
            self.user_attrs = self.sgd_lin_optuner.user_attrs

            print(f"\n\nFinished training. TOP {self.select_top} models are:")
            display(self.sgd_lin_lb.head(self.select_top))
            print(384 * "-", "\n\n")




        # Train meta-learner if possible
        if self.train_meta and self.dataset.preprocessor.val_folds and not self.dataset.preprocessor.forecasting:
            print("TRAINING ELASTICNET META-LEARNER ...\n")
            print("Retraining best found models on each fold and generating oof preds ...")
            self._train_best_on_folds()
            print("Training meta-learner using Optuna...")
            self._train_meta_learner()
            print(384 * "-", "\n\n")
        
        
        # Retraining best found models on entire train dataset
        print("REFITTING BEST FOUND MODELS ON ENTIRE TRAIN DATASET ...\n")
        self._train_best_on_full_train_data()
        print("Done.\n")
        print(384 * "-", "\n\n")

        # Generating and printing all optuned models LEADERBOARD
        print("\nLEADERBOARD:\n\n")
        self.leaderboard = self.get_leaderboard()   
        display(self.leaderboard)
        print(384 * "-", "\n\n")

        # Ensemble model
        print("Generated ensemble model. To use it, call .predict() / .predict_proba() method on trainer object.")
        self.ensemble = self.get_ensemble()
        display(self.ensemble)
        print(384 * "-", "\n\n")


        # Generating and displaying final evaluation on eval_dataset
        print("External Evaluation LEADERBOARD:") 
        self.X_eval, self.y_eval = self.eval_dataset.process()
        self.eval_lb = self.evaluate(self.X_eval, self.y_eval)
        display(self.eval_lb)
        print(384 * "-", "\n\n")
        self.best_model = self.eval_lb.iloc[0]["model"]

        # Retraining best found models on whole available data, saving trainer
        print("Retraining best found models on whole available data...")
        self._train_best_on_whole_data()
        self.dump(self.save_path)
        print(f"Done. Trainer is ready for inference and saved at path: {self.save_path}")
        print(384 * "-", "\n\n")





    def _train_best_on_folds(self):
        """
        Retrains best (select_top) models of each model type on each fold.
        Generates OOF predictions for each model to use with the meta learner.
        """
        self.meta_dict = {}

        # Add models as keys with initial value None using ID column
        for df in [self.xgb_lb, self.lgb_lb, self.cat_lb, self.rf_lb, self.sgd_lin_lb]:
            if df is not None:
                for _, row in df.head(self.select_top).iterrows():
                    model_id = row["id"]  # Use the ID column value
                    self.meta_dict[model_id] = {"model": row["model"], "oof_preds": None}

        # Fit models and store OOF predictions and ground truth labels
        for model_id, model_info in self.meta_dict.items():
            model = model_info["model"]
            all_oof_preds = []
            all_ground_truths = []
            for fold in self.processed_data["data"]:
                # Fit model on training data
                model.fit(fold["train"][0], fold["train"][1])
                
                # Predict on validation data
                if self.dataset.preprocessor.prob_type != "regression":
                    # pred_method = (model.predict_proba if hasattr(model, "predict_proba") else model.predict)
                    oof_preds = model.predict_proba(fold["val"][0]).squeeze()
                else:
                    oof_preds = model.predict(fold["val"][0]).squeeze()
                all_oof_preds.append(oof_preds)
                
                # Collect ground truth labels from validation data
                ground_truths_fold = fold["val"][1]
                all_ground_truths.append(ground_truths_fold)
            
            # Store OOF predictions and ground truth labels in the dictionaries
            self.meta_dict[model_id]["oof_preds"] = {
                "predictions": np.concatenate(all_oof_preds),  # Combine OOF predictions
                "ground_truths": np.concatenate(all_ground_truths)  # Combine ground truth labels
            }
            
            


    def _train_meta_learner(self):


        """
        Creates and trains an ElasticNet meta-learner using out-of-fold probabilities from various GBMs models.
        Uses Optuna for hyperparameter optimization.
        """

        
        # Initialize list to collect predictions and ground truth labels
        all_preds = []
        ground_truth = None

        # Extract predictions and ground truth from the meta_dict dictionary
        for model_key, data in self.meta_dict.items():
            preds = data["oof_preds"]["predictions"]
            labels = data["oof_preds"]["ground_truths"]

            # Set ground truth labels from the first model (since they are the same across models)
            if ground_truth is None:
                ground_truth = labels
            
            all_preds.append(preds)

        # Ensure all predictions are 2D arrays (n_samples, 1) before stacking
        all_preds = [preds.reshape(-1, 1) if preds.ndim == 1 else preds for preds in all_preds]
        
        # Convert lists to numpy arrays
        X = np.hstack(all_preds)  # Shape (n_samples, n_models * n_predicted_values)
        y = ground_truth  # Shape (n_samples,)

        # Scorer
        meta_scorer = make_scorer(self.eval_metric.score, 
                                  greater_is_better=self.eval_metric.greater_is_better)

        def objective(trial):
            # Define the hyperparameters to optimize
            alpha = trial.suggest_float('alpha', 1e-6, 100.0, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            max_iter = trial.suggest_int('max_iter', 1000, 20000, log=True)

            # Create an ElasticNet model
            model = ElasticNet(alpha=alpha, 
                            l1_ratio=l1_ratio, 
                            max_iter=max_iter,
                            random_state=self.seed)

            # Perform cross-validation
            warnings.simplefilter('ignore', sklearn.exceptions.ConvergenceWarning)
            score = cross_val_score(model, X, y, cv=self.dataset.preprocessor.n_folds, scoring=meta_scorer, 
                                    # n_jobs=-1
                                    )

            # Return the mean score
            return score.mean()

        # Create a study object and optimize the objective function
        self.meta_study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(multivariate=True),
            pruner=optuna.pruners.MedianPruner()
        )
        self.meta_study.optimize(objective, timeout=self.meta_timeout)

        # Get the best parameters
        best_params = self.meta_study.best_params

        # Train the final model with the best parameters
        final_model = ElasticNet(alpha=best_params['alpha'], 
                                l1_ratio=best_params['l1_ratio'], 
                                max_iter=best_params['max_iter'], 
                                random_state=self.seed)
        final_model.fit(X, y)

        print("Done:")
        print(f"Number of models tried by Optuna: {len(self.meta_study.trials)}.")
        print("Best ElasticNet parameters:", best_params)
        print(f"Best {self.eval_metric.name} score:", self.meta_study.best_value)

        self.meta_learner = final_model





    # def _train_meta_learner(self):
    # 
    #     """
    #     Creates and trains an ElasticNetCV meta-learner using out-of-fold probabilities from various GBMs models.
    #     Then refits the best ElasticNet model (without cross-validation) on the full dataset.
    #     """
        
    #     # Initialize list to collect predictions and ground truth labels
    #     all_preds = []
    #     ground_truth = None

    #     # Extract predictions and ground truth from the meta_dict dictionary
    #     for model_key, data in self.meta_dict.items():
    #         preds = data["oof_preds"]["predictions"]
    #         labels = data["oof_preds"]["ground_truths"]

    #         # Set ground truth labels from the first model (since they are the same across models)
    #         if ground_truth is None:
    #             ground_truth = labels
            
    #         all_preds.append(preds)

    #     # Ensure all predictions are 2D arrays (n_samples, 1) before stacking
    #     all_preds = [preds.reshape(-1, 1) if preds.ndim == 1 else preds for preds in all_preds]
        
    #     # Convert lists to numpy arrays
    #     X = np.hstack(all_preds)  # Shape (n_samples, n_models * n_predicted_values)
    #     y = ground_truth  # Shape (n_samples,)

    #     # Train ElasticNetCV with cross-validation to find the best parameters
    #     elasticnet_cv = ElasticNetCV(
    #         l1_ratio=np.linspace(1e-5, 1.0, 500),
    #         n_alphas=1000,  # Number of alphas to try
    #         cv=self.dataset.preprocessor.n_folds,  # Cross-validation folds
    #         max_iter=10000,
    #         random_state=self.seed,
    #         # n_jobs=-1  # Parallelization
    #     )
        
    #     # Fit the ElasticNetCV model
    #     elasticnet_cv.fit(X, y)

        
    #     # Refit ElasticNet model using the best-found parameters, but without CV
    #     final_model = ElasticNet(
    #         alpha=elasticnet_cv.alpha_,
    #         l1_ratio=elasticnet_cv.l1_ratio_,
    #         max_iter=10000,
    #         random_state=self.seed
    #     )

    #     # Scorer
    #     meta_scorer = make_scorer(self.eval_metric.score, 
    #                               greater_is_better=self.eval_metric.greater_is_better)

    #     # Calculate mean cross-validated score using a desired scoring metric
    #     mean_cv_score = cross_val_score(final_model, X, y, cv=self.dataset.preprocessor.n_folds, scoring=meta_scorer, n_jobs=-1).mean()

    #     # Print cross-validated results
    #     print("Done:")
    #     print(f"Best alpha: {elasticnet_cv.alpha_}")
    #     print(f"Best l1_ratio: {elasticnet_cv.l1_ratio_}")
    #     print(f"Mean cross-validation {self.eval_metric.name} score:", mean_cv_score)

        
    #     # Refit on the entire dataset without cross-validation
    #     final_model.fit(X, y)

    #     # Store the final trained model
    #     self.meta_learner = final_model



    
    def predict_meta(self, X):
        """
        Makes predictions using the trained meta-learner.
        
        Args:
        X (array-like): The input data to predict on.
        
        Returns:
        array-like: The predictions from the meta-learner.
        """

        # Get predictions from all base models
        all_preds = []
        for model_id in self.meta_dict.keys():
            model = self.meta_dict[model_id]["model"]
            if self.dataset.preprocessor.prob_type != "regression":
                # pred_method = (model.predict_proba if hasattr(model, "predict_proba") else model.predict)
                preds = model.predict_proba(X)
            else:
                preds = model.predict(X).reshape(-1, 1)
            all_preds.append(preds)

        # Combine predictions
        X_meta = np.hstack(all_preds)

        # Make final prediction using meta-learner
        final_predictions = self.meta_learner.predict(X_meta)
        if self.dataset.preprocessor.prob_type != "regression":
            final_predictions = np.round(final_predictions)
        return final_predictions




    def _train_best_on_full_train_data(self):

        """
        Retrains best (select_top) models of each model type on the entire train dataset.
        Updates the models in the leaderboards and the meta_dict dictionary.
        """

        self.X_train_full, self.y_train_full = self.dataset.preprocessor.transform(self.dataset.df, fit=True)
    
        
        self.all_lbs = [self.xgb_lb, self.lgb_lb, self.cat_lb, self.rf_lb, self.sgd_lin_lb]
        
        for leaderboard in self.all_lbs:
            if leaderboard is not None:
                df = leaderboard.head(self.select_top)
                
                for idx, row in df.iterrows():
                    # Retrain the model
                    retrained_model = row['model'].fit(self.X_train_full, 
                                                       self.y_train_full)
                    
                    # Update the model in the leaderboard
                    df.at[idx, 'model'] = retrained_model
                    
                    # Update meta_dict with retrained model
                    model_id = row["id"]
                    if model_id in self.meta_dict:
                        self.meta_dict[model_id]["model"] = retrained_model


        
    def _train_best_on_whole_data(self):

        """
        Retrains best (select_top) models of each model type on the entire available data.
        This includes both train and external eval datasets.
        Updates the models in the global leaderboard
        """

        self.whole_df = pd.concat([self.dataset.df, self.eval_dataset.df], axis=0)
        self.X_train_whole, self.y_train_whole = self.dataset.preprocessor.transform(self.whole_df, fit=True)     
                
        for idx, row in self.leaderboard.iterrows():
            # Retrain the model
            retrained_model = row['model'].fit(self.X_train_whole, 
                                                self.y_train_whole)
            
            # Update the model in the leaderboard
            self.leaderboard.at[idx, 'model'] = retrained_model
            
            # Update meta_dict with retrained model
            model_id = row["id"]
            if model_id in self.meta_dict:
                self.meta_dict[model_id]["model"] = retrained_model
        
        self.ensemble = self.get_ensemble() # replace VotingModel estimators with retrained base models




    def get_leaderboard(self) -> pd.DataFrame:

        """
        
        Generates a global leaderboard dataframe. Collects best models of each type from model leaderboards,
        alongside their hyperparams.
        
        """

        # Select relevant columns only
        cols = ["id"] + self.user_attrs + ["model"]

        # Collect the model DataFrames that are not None
        leaderboards = [x for x in [self.lgb_lb, self.cat_lb, self.xgb_lb, self.rf_lb, self.sgd_lin_lb] if x is not None]

        # Apply slicing and column selection
        leaderboards = [df.loc[:self.select_top-1, cols] for df in leaderboards]

        # Concatenate the DataFrames
        leaderboard = pd.concat(leaderboards, ignore_index = True)

        # Sort the concatenated DataFrame
        leaderboard = leaderboard.sort_values(by = self.user_attrs[-1], # last logged metric is optimized
                                              ascending = not(self.eval_metric.greater_is_better), 
                                              ignore_index = True)
        
        return leaderboard
    


    
    def get_ensemble(self) -> Union[VotingRegressor, VotingClassifier]:

        """
        
        Generates and returns a Sklearn VotingRegressor / VotingClasssifier ensemble model
        from top models, as found by Optuna.

        """

        estimators = list(zip(self.leaderboard.id.values, self.leaderboard.model.values))

        if self.eval_metric.greater_is_better:
            weights = self.leaderboard[self.user_attrs[-1]] / sum(self.leaderboard[self.user_attrs[-1]]) 
        else:
            # inverse for losses, to give higher weights to models with lower loss
            weights = 1 / self.leaderboard[self.user_attrs[-1]] / sum(1 / self.leaderboard[self.user_attrs[-1]]) 

        # Handle both tasks types   

        if self.pb_type == "regression":

            ensemble_model = VotingRegressor(estimators = estimators, 
                                             weights = weights)
        else:
            ensemble_model = VotingClassifier(estimators = estimators, 
                                              weights = weights, 
                                              voting = "soft")
            

        # Adding additional class params to mark model as fitted
            ensemble_model.le_ = self.dataset.preprocessor.label_encoder
            ensemble_model.classes_ = ensemble_model.le_.classes_
            
        ensemble_model.estimators_ = self.leaderboard.model.values
        ensemble_model.named_estimators_ = Bunch()

        for name, est in estimators:
            ensemble_model.named_estimators_[name] = est

        ensemble_model.__dict__["n_features_in_"] = len(self.processed_data["data"][0]["train"][0].iloc[0])
        ensemble_model.__dict__["feature_names_in_"] = np.array(list(self.processed_data["data"][0]["train"][0].columns))

        ensemble_model._predict = custom_predict.__get__(ensemble_model)

        return ensemble_model
    
    



    def compute_metrics(self,
                        model,
                        X: Union[pd.DataFrame, np.ndarray],
                        y: Union[np.ndarray, list]) -> dict:
        
        """

        Computes various metrics based on the model predictions and true labels.

        Parameters:
        - model: The trained model.
        - X: Input features for prediction.
        - y: True labels for evaluation.

        Returns:
        - dict: A dictionary containing computed metrics.
        
        """


        # Get predictions from the model
        preds = self.predict(X, mode=model)

        # Initialize an empty dictionary to store metrics
        metrics_dict = {}

        # Check if the problem type is regression

        if self.dataset.preprocessor.prob_type == "regression":

            # Compute regression metrics
            
            mae = mean_absolute_error(y, preds)
            mse = mean_squared_error(y, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, preds)
            custom_metric = self.eval_metric.score(y, preds)

            # Populate the metrics dictionary for regression
            metrics_dict = {
                "custom_" + self.eval_metric.name: custom_metric,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
            }
        
        else:  # If the problem type is classification

            # Handle specific cases based on problem type and model type

            preds = self.dataset.preprocessor.label_encoder.transform(preds) # re-ecode labels, decoded by self.predict()
            
            # Compute metrics based on problem type

            if self.dataset.preprocessor.prob_type == "binary":

                # Binary classification metrics
                accuracy = accuracy_score(y, preds)
                precision, recall, f1, _ = precision_recall_fscore_support(y, preds, 
                                                                           average = "weighted",
                                                                           zero_division = 0)
                custom_metric = self.eval_metric.score(y, preds)

            elif self.dataset.preprocessor.prob_type == "multiclass":

                # Multi-class classification metrics
                accuracy = accuracy_score(y, preds)
                precision, recall, f1, _ = precision_recall_fscore_support(y, preds,
                                                                           average = "weighted",
                                                                           zero_division = 0)
                custom_metric = self.eval_metric.score(y, preds)

            # Populate the metrics dictionary for classification
            metrics_dict = {
                "custom_" + self.eval_metric.name: custom_metric,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        return metrics_dict
 



    def evaluate(self,
                  X: Union[pd.DataFrame, np.ndarray],
                  y: Union[np.ndarray, list]):
        
        """
        Evaluate models in leaderboard and ensemble on the eval dataset.

        Parameters:
        - X (Union[pd.DataFrame, np.ndarray]): Features for evaluation.
        - y (Union[np.ndarray, list]): True labels for evaluation.

        Returns:
        - pd.DataFrame: DataFrame with model IDs, evaluation metrics, sorted by the evaluation metric.

        """
        
        ids = self.leaderboard["id"].to_list() + ["VotingEnsemble"] + ["MetaLearner"]
        models = self.leaderboard["model"].to_list() + [self.ensemble, self.meta_learner]
        eval_df = pd.DataFrame({
            "id": ids,
            "model": models
        })

        metrics_df = eval_df["model"].apply(lambda model: pd.Series(self.compute_metrics(model, X, y)))

        return pd.concat([eval_df, metrics_df], axis=1).sort_values(by = "custom_" + self.eval_metric.name,
                                                                         ascending = not(self.eval_metric.greater_is_better),
                                                                         ignore_index = True)
    
    

    def predict(self, 
                X: Union[pd.DataFrame, np.ndarray, list],
                mode = "ensemble") -> np.ndarray:
        
        """

        Calling .predict() method will generate model predictions from input data X.

        Make sure data has same format as seen during fitting.

        Output prediction will have shape (n_features,). It will consist only of labels
        if task is classification.
        
        Parameters:

            X (Union[pd.Dataframe, np.ndarray, list]): The input data features to make 
                                                       predictions on.

            mode (str): Prediction mode. Can be either "ensemble" or custom model "id" as 
                        seen in leaderboard. Default is "ensemble".

        """

        
        if self.ensemble is None:
            raise Exception("TabularAML Trainer isn't fitted.")
        
        # Handle prediction mode type (from "ensemble", "meta" or specific model "id")
        if mode == "ensemble":
            model = self.ensemble
        elif mode == "meta":
            model = self.meta_learner
        elif mode in self.leaderboard["id"].to_list():
            model = self.leaderboard.loc[self.leaderboard["id"] == mode].iloc[0]["model"]
        elif isinstance(mode, tuple(possible_models)):
            model = mode
        else:
            raise Exception("Invalid mode. It should be either 'ensemble', or a model ID or model from leaderboard.")
        
        
        # Predict. Make sure we return true labels
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                if mode == "meta" or mode == self.meta_learner:
                    preds = self.predict_meta(X)
                else:
                    preds = model.predict(X)

        # Inverse-transform encoded labels for classification if model is not VotingClassifier (which handles that internally)
        if mode != "ensemble" and not(isinstance(model, VotingClassifier)) and self.dataset.preprocessor.prob_type != "regression":
            preds = self.dataset.preprocessor.label_encoder.inverse_transform(preds.squeeze()) # squeeze last axis for CAT

        return preds
    
    

    def predict_proba(self, 
                      X: Union[pd.DataFrame, np.ndarray, list],
                      mode = "ensemble") -> np.ndarray:
           
        """

        Calling .predict_proba() method will generate the model's classification
        probabilities distribution from input data X. Only works if task is "binary"/
        "mutliclass".

        Make sure data has same format as seen during fitting.

        Output prediction will have shape (n_features, n_classes).
        
        Parameters:

            X (Union[pd.Dataframe, np.ndarray, list]): The input data features to make 
                                                       predictions on.

            mode (str): Prediction mode. Can be either "ensemble" or custom model "id" as 
                        seen in leaderboard. Default is "ensemble".

        """

        # Handle task type

        if self.ensemble is None:
            raise Exception("TabularAML Trainer isn't fitted.")
        if self.pb_type == "regression":
            raise Exception("Can't generate probs distribution for regression task.")
        if mode == "ensemble":
            model = self.ensemble
        elif mode in self.leaderboard["id"].values:
            model = self.leaderboard.loc[self.leaderboard["id"] == mode].iloc[0]["model"]
        else:
            raise Exception("Invalid mode. It should be either 'ensemble', or a model ID from leaderboard.")
        
        # Predict proba

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                preds = model.predict_proba(X)

        return preds
    

    def dump(self, file_path):
        with open(file_path, "wb") as f:
            dill.dump(self, f)

    def load(self, file_path):
        with open(file_path, "rb") as f:
            self = dill.load(f)