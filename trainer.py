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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold
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
from sklearn.preprocessing import MinMaxScaler


class RankAveragingEnsemble:
    def __init__(self, estimators):
        self.estimators = estimators
    def predict(self, X):
        # Get predictions from each estimator
        all_preds = [model.predict(X) for _, model in self.estimators]
        # Convert each set of predictions to rank values
        ranked_preds = np.array([rankdata(pred) for pred in all_preds])   
        # Return the sum of ranks across models (instead of the mean)
        return np.sum(ranked_preds, axis=0)
    
    
possible_models = [LGBMRegressor, LGBMClassifier,
                   CatBoostRegressor, CatBoostClassifier,
                   XGBRegressor, XGBClassifier,
                   RandomForestRegressor, RandomForestClassifier,
                   SGDRegressor, SGDClassifier,
                   VotingRegressor, RankAveragingEnsemble, VotingClassifier,
                   ElasticNet, ElasticNetCV, LogisticRegression, Ridge]


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', sklearn.exceptions.ConvergenceWarning)
import pandas as pd
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import numpy as np
from scipy.stats import rankdata


import os
import dill
from joblib import Parallel, delayed
import re
import gc
from typing import Union, Dict, Optional, Tuple, Literal
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
                 ensemble_method: Optional[Literal["weighted", "mean", "rank"]] = "weighted",
                 save_path = None):
        
        
        """

            TabularAML Trainer. Designed to optimize, find, train and ensemble best model
        for your tabular data. Used in both regression and binary / multiclass tasks.

            Parameters:

                * dataset (TabularDataset): A TabularAML TabularDataset instance with your data.
                Used in CV hyperparam-tuning (with early stopping) and for refitting models at end on
                the entire data.

                * eval_dataset (TabularDataset): A TabularDataset used exclusively for evaluation
                purposes: generating final leaderboard (including Ensemble model(s)).

                * eval_metric (Scorer | str): A Scorer instance with the eval metric chosen to be used 
                                            during training for optimization (early stopping) and for evaluation, logging and
                                            ranking purposes.
                                            Can be an abbreviation of a common loss / score, such as "rmse", "rmsle",
                                            "mae", "mse", "r2". 
                                            If no eval_metric is provided, "rmse" will be used by default.  


                * early_stopping_rounds (int): Stops training after specified rounds without validation improvement. 
                                             Used for faster training, allowing a deeper exploration of the search space,
                                             possibly at the expense of full reproducibility of validation performance
                                             on the test set.
                                             WARNING: Using this parameter, especially with low non-zero values,
                                                      may cause underfitting and poor reproducibility.
                                                      Consider using `of_mitigation_level` instead for more stable performance.
                                                      Default: 0 (disabled).

                * of_mitigation_level (float): Recommended way to prevent overfitting, while also avoiding underfitting.
                                             Penalizes train-validation metric gap during Optuna optimization.
                                             Helps control overfitting without affecting model depth.
                                             Use with disabled early stopping for more consistent results.
                                             Default: 0.2.

                * models (list): List of models to use. Select from:
                                    * "LGB" for LGBMRegressor / LGBMClassifier
                                    * "CAT" for CatBoostRegressor / CatBoostClassifier
                                    * "XGB" for XGBRegressor / XGBClassifer
                                    * "RF" for RandomForestRegressor / RandomForestClassifier
                                    * "SGD_LINEAR" for SGDRegressor / SGDClassifier
                                Default is ["LGB", "XGB", "CAT", "RF", "SGD_LINEAR"].

                * hyperparams (dict): A dictionary of dictionaries with default hyperparameter search space for each model.
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
                
                * use_gpu (bool): Whether to use any GPU for training.
                                  Default is True.

                * use_cuda (bool): Whether to use Nvidia CUDA GPU training.
                                   Defaults to True.

                * n_trials (int): Number of trials to perform in the Optuna Study.
                                  Default is 10000 (as high as possible).

                * timeout (int): Number of seconds after which all GBMs models tuning stops automatically.
                                 Default is 3000s or 50 mins.

                * seed (int): Random state seed to use for training the model. It ensures a level of reproducibility.
                              Default is 42.

                * select_top (int): Select best x models for each model type, as determined by Optuna.
                                    Default is 3.

                * train_meta (bool): Whether to train meta-model on top of the best models of each kind.
                                     Default is True.

                * meta_timeout (int): Number of seconds after which meta model study stops automatically.
                                      Default is 600s or 10 mins.

                * ensemble_method (str): Method for ensembling top models. Choose from:
                                        * "weighted": Weighted averaging based on model performance.
                                        * "mean": Simple mean averaging of model predictions.
                                        * "rank": Rank-based averaging, where model predictions are ranked and averaged.
                                        Default is "weighted" ("mean"?).

                * save_path (str): Path to save the trainer instance to. File extension needs to be .pkl.
                                   If not set, no saving will occur.
                

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
        self.ensemble_method = ensemble_method
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
        self.meta_scaler = None

        
        
    
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
        if self.save_path is not None:
            self.dump(self.save_path)
        print(f"Done. Trainer is ready for inference and saved at path: {self.save_path}")
        print(384 * "-", "\n\n")




    def _train_best_on_folds(self):
        """
        Retrains best (select_top) models of each model type on each fold.
        Generates OOF predictions for each model to use with the meta learner.
        """
        self.meta_dict = {}

        # Initialize meta_dict with top models
        for lb in [self.xgb_lb, self.lgb_lb, self.cat_lb, self.rf_lb, self.sgd_lin_lb]:
            if lb is not None:
                for _, row in lb.head(self.select_top).iterrows():
                    model_id = row["id"]
                    self.meta_dict[model_id] = {
                        "model": row["model"],
                        "oof_preds": None
                    }

        # Loop folds and collect OOF predictions + ground truths
        for model_id, info in self.meta_dict.items():
            model = info["model"]
            all_oof = []
            all_truth = []

            for fold in self.processed_data["data"]:
                X_tr, y_tr = fold["train"]
                X_val, y_val = fold["val"]

                # fit on fold
                model.fit(X_tr, y_tr)

                # predict
                if self.dataset.preprocessor.prob_type != "regression":
                    preds = model.predict_proba(X_val)
                    # if you only need one-class proba, you could e.g. preds[:, 1]
                else:
                    preds = model.predict(X_val).reshape(-1, 1)

                # wrap into DataFrame/Series, resetting index
                pred_df = pd.DataFrame(preds).reset_index(drop=True)
                truth_ser = (
                    y_val.reset_index(drop=True)
                    if hasattr(y_val, "reset_index")
                    else pd.Series(y_val).reset_index(drop=True)
                )

                all_oof.append(pred_df)
                all_truth.append(truth_ser)

            # concatenate across folds
            combined_oof = pd.concat(all_oof, ignore_index=True)
            combined_truth = pd.concat(all_truth, ignore_index=True)

            self.meta_dict[model_id]["oof_preds"] = {
                "predictions": combined_oof,
                "ground_truths": combined_truth
            }


    def _train_meta_learner(self):
        """
        Creates and trains an ElasticNet meta-learner using scaled out-of-fold predictions.
        Scales OOF predictions using MinMaxScaler.
        Uses Optuna for hyperparameter optimization and prints detailed training information.
        """
        # Initialize list to collect predictions and ground truth labels
        all_preds = []
        ground_truth = None

        # Extract predictions and ground truth from the meta_dict dictionary
        for model_key, data in self.meta_dict.items():
            preds = data["oof_preds"]["predictions"]
            labels = data["oof_preds"]["ground_truths"]

            # Set ground truth labels from the first model (since they are consistent across models)
            if ground_truth is None:
                ground_truth = labels

            all_preds.append(preds)

        # Ensure predictions are aligned by index and combine them
        X = pd.concat(all_preds, axis=1)
        y = ground_truth

        # Scale predictions using MinMaxScaler
        self.meta_scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(self.meta_scaler.fit_transform(X), index=X.index, columns=X.columns)

        # Scorer
        meta_scorer = self.eval_metric.score

        def objective(trial):
            alpha = trial.suggest_float('alpha', 1e-6, 100.0, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            max_iter = trial.suggest_int('max_iter', 1000, 20000, log=True)

            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                random_state=self.seed
            )

            # Cross-validation
            cv = KFold(n_splits=self.dataset.preprocessor.n_folds, shuffle=True, random_state=self.seed)

            # Function to compute train and validation scores
            def process_split(train_idx, val_idx):
                X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx] #update
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)

                # ElasticNet outputs continuous values; convert to class labels for classification
                if self.pb_type != "regression":
                    n_classes = len(np.unique(y))
                    y_train_pred = np.clip(np.round(y_train_pred), 0, n_classes - 1).astype(int)
                    y_val_pred = np.clip(np.round(y_val_pred), 0, n_classes - 1).astype(int)

                train_score = meta_scorer(y_train, y_train_pred)
                val_score = meta_scorer(y_val, y_val_pred)

                return train_score, val_score

            results = Parallel(n_jobs=-1)(
                delayed(process_split)(train_idx, val_idx) for train_idx, val_idx in cv.split(X_scaled, y)
            )

            train_scores, val_scores = zip(*results)
            trial.set_user_attr("mean_train_score", np.mean(train_scores))
            return np.mean(val_scores)

        # Optimize using Optuna
        self.meta_study = optuna.create_study(
            direction='maximize' if self.eval_metric.greater_is_better else "minimize",
            sampler=optuna.samplers.TPESampler(multivariate=True),
            pruner=optuna.pruners.MedianPruner()
        )
        self.meta_study.optimize(objective, timeout=self.meta_timeout)

        # Get best parameters and scores
        best_params = self.meta_study.best_params
        best_trial = self.meta_study.best_trial
        mean_train_score = best_trial.user_attrs["mean_train_score"]
        mean_val_score = self.meta_study.best_value

        # Train final model with best parameters
        self.meta_learner = ElasticNet(**best_params, random_state=self.seed)
        self.meta_learner.fit(X_scaled, y)

        # Final prints
        print("Done:")
        print(f"Number of models tried by Optuna: {len(self.meta_study.trials)}.")
        print("Best ElasticNet parameters:", best_params)
        print(f"Best mean train {self.eval_metric.name} score:", mean_train_score)
        print(f"Best mean val {self.eval_metric.name} score:", mean_val_score)




    def predict_meta(self, X):
        """
        Makes predictions using the trained meta-learner with scaled input probabilities.
        """
        all_preds = []
        for model_id in self.meta_dict.keys():
            model = self.meta_dict[model_id]["model"]
            if self.dataset.preprocessor.prob_type != "regression":
                preds = model.predict_proba(X)
            else:
                preds = model.predict(X).reshape(-1, 1)
            all_preds.append(preds)

        # Combine predictions and scale them using the stored scaler
        X_meta = np.hstack(all_preds)
        X_meta_scaled = self.meta_scaler.transform(X_meta)

        # Make final prediction using meta-learner
        final_predictions = self.meta_learner.predict(X_meta_scaled)
        if self.dataset.preprocessor.prob_type != "regression":
            n_classes = len(self.dataset.preprocessor.label_encoder.classes_)
            final_predictions = np.clip(np.round(final_predictions), 0, n_classes - 1).astype(int)
        return final_predictions
        


    def _train_best_on_full_train_data(self):
        """
        Retrains best (select_top) models of each model type on the entire train dataset.
        Updates the models in the leaderboards and the meta_dict dictionary.
        """
        # get full train
        self.X_train_full, self.y_train_full = (
            self.dataset.preprocessor.transform(self.dataset.df, fit=True)
        )

        for lb in [self.xgb_lb, self.lgb_lb, self.cat_lb, self.rf_lb, self.sgd_lin_lb]:
            if lb is not None:
                top_df = lb.head(self.select_top)
                for idx, row in top_df.iterrows():
                    retrained = row["model"].fit(self.X_train_full, self.y_train_full)
                    top_df.at[idx, "model"] = retrained

                    # sync into meta_dict
                    model_id = row["id"]
                    if model_id in self.meta_dict:
                        self.meta_dict[model_id]["model"] = retrained

                # write back updated models
                lb.update(top_df)

    
    def _train_best_on_whole_data(self):
        """
        Retrains best (select_top) models of each model type on the entire available data.
        This includes both train and external eval datasets.
        Updates the models in the global leaderboard.
        """
        # combine train + eval
        self.whole_df = pd.concat([self.dataset.df, self.eval_dataset.df], axis=0)
        self.X_train_whole, self.y_train_whole = (
            self.dataset.preprocessor.transform(self.whole_df, fit=True)
        )

        # retrain every model in the global leaderboard
        for idx, row in self.leaderboard.iterrows():
            retrained = row["model"].fit(self.X_train_whole, self.y_train_whole)
            self.leaderboard.at[idx, "model"] = retrained

            # sync into meta_dict
            model_id = row["id"]
            if model_id in self.meta_dict:
                self.meta_dict[model_id]["model"] = retrained

        # rebuild voting ensemble with retrained base models
        self.ensemble = self.get_ensemble()


        
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
    
    



    def get_ensemble(self):
        """
        Generates and returns an ensemble model based on the specified ensemble method.
        For "rank" method, performs rank-averaging instead of using VotingRegressor/Classifier.
        Otherwise, reintroduces the older code to create and augment the Voting ensemble.
        """
        estimators = list(zip(self.leaderboard.id.values, self.leaderboard.model.values))

        if self.ensemble_method == "rank":
            # --- RANK LOGIC (short-circuits here) ---
            return RankAveragingEnsemble(estimators)
        else:
            # --- For "weighted" or "mean", reintroduce your old snippet. ---

            # 1) Compute weights if "weighted"
            #    If "mean", weights will be None, leading to uniform weighting
            if self.ensemble_method == "weighted":
                if self.eval_metric.greater_is_better:
                    weights = (
                        self.leaderboard[self.user_attrs[-1]]
                        / sum(self.leaderboard[self.user_attrs[-1]])
                    )
                else:
                    # Invert for lower-is-better metrics
                    weights = (
                        1 / self.leaderboard[self.user_attrs[-1]]
                        / sum(1 / self.leaderboard[self.user_attrs[-1]])
                    )
            elif self.ensemble_method == "mean":
                weights = None
            else:
                raise ValueError("Invalid ensemble_method. Choose from 'weighted', 'mean', or 'rank'.")

            # 2) Create the voting ensemble model
            if self.pb_type == "regression":
                ensemble_model = VotingRegressor(estimators=estimators, weights=weights)
            else:
                ensemble_model = VotingClassifier(estimators=estimators, weights=weights, voting="soft")

                # Adding additional class params to mark model as fitted
                ensemble_model.le_ = self.dataset.preprocessor.label_encoder
                ensemble_model.classes_ = ensemble_model.le_.classes_

            # 3) Set the attributes as in your old code
            ensemble_model.estimators_ = self.leaderboard.model.values
            ensemble_model.named_estimators_ = Bunch()
            for name, est in estimators:
                ensemble_model.named_estimators_[name] = est

            # 4) Optional: Store feature information if you want to mark them as fitted
            #    (Adapt indices and columns for your actual data)
            ensemble_model.__dict__["n_features_in_"] = len(
                self.processed_data["data"][0]["train"][0].iloc[0]
            )
            ensemble_model.__dict__["feature_names_in_"] = np.array(
                list(self.processed_data["data"][0]["train"][0].columns)
            )

            # 5) Attach a custom_predict if needed
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
            rmse = root_mean_squared_error(y, preds)
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
    

    def dump(self, path: str) -> None:
        """
        Save the entire trainer state, including:
        - Initialization arguments (models, hyperparams, etc.)
        - Fitted models and leaderboards
        - Preprocessing or meta-learner details
        """
        # Put whatever you need for re-loading the trainer into a dictionary:
        state = {
            # Basic init args
            "dataset": self.dataset,  # includes self.dataset.preprocessor
            "eval_dataset": self.eval_dataset,
            "eval_metric": self.eval_metric,
            "early_stopping_rounds": self.early_stopping_rounds,
            "of_mitigation_level": self.of_mitigation_level,
            "models": self.models,
            "hyperparams": self.hyperparams,
            "use_gpu": self.use_gpu,
            "use_cuda": self.use_cuda,
            "n_trials": self.n_trials,
            "timeout": self.timeout,
            "seed": self.seed,
            "select_top": self.select_top,
            "train_meta": self.train_meta,
            "meta_timeout": self.meta_timeout,
            "ensemble_method": self.ensemble_method,
            "save_path": self.save_path,

            # Internal trainer attributes from training
            "processed_data": self.processed_data,
            "mode": self.mode,
            "pb_type": self.pb_type,
            "user_attrs": self.user_attrs,

            "lgb_lb": self.lgb_lb,
            "cat_lb": self.cat_lb,
            "xgb_lb": self.xgb_lb,
            "rf_lb": self.rf_lb,
            "sgd_lin_lb": self.sgd_lin_lb,
            "leaderboard": self.leaderboard,
            "eval_lb": self.eval_lb,
            "ensemble": self.ensemble,
            "best_model": self.best_model,

            "X_train_full": self.X_train_full,
            "y_train_full": self.y_train_full,
            "X_eval": self.X_eval,
            "y_eval": self.y_eval,

            "meta_study": self.meta_study,
            "meta_dict": self.meta_dict,
            "meta_learner": self.meta_learner,
            "meta_scaler": self.meta_scaler,
        }

        with open(path, "wb") as f:
            dill.dump(state, f)
        print(f"Trainer successfully dumped to {path}.")


    @classmethod
    def load(cls, path: str):
        """
        Load an entire Trainer from disk, restoring all
        attributes such that it can be used for inference
        or further usage.
        """
        with open(path, "rb") as f:
            state = dill.load(f)

        # Re-create the Trainer with the same init arguments
        trainer = cls(
            dataset=state["dataset"],
            eval_dataset=state["eval_dataset"],
            eval_metric=state["eval_metric"],
            early_stopping_rounds=state["early_stopping_rounds"],
            of_mitigation_level=state["of_mitigation_level"],
            models=state["models"],
            hyperparams=state["hyperparams"],
            use_gpu=state["use_gpu"],
            use_cuda=state["use_cuda"],
            n_trials=state["n_trials"],
            timeout=state["timeout"],
            seed=state["seed"],
            select_top=state["select_top"],
            train_meta=state["train_meta"],
            meta_timeout=state["meta_timeout"],
            ensemble_method=state["ensemble_method"],
            save_path=state["save_path"],
        )

        # Restore any internal trainer fields
        trainer.processed_data = state["processed_data"]
        trainer.mode = state["mode"]
        trainer.pb_type = state["pb_type"]
        trainer.user_attrs = state["user_attrs"]

        trainer.lgb_lb = state["lgb_lb"]
        trainer.cat_lb = state["cat_lb"]
        trainer.xgb_lb = state["xgb_lb"]
        trainer.rf_lb = state["rf_lb"]
        trainer.sgd_lin_lb = state["sgd_lin_lb"]
        trainer.leaderboard = state["leaderboard"]
        trainer.eval_lb = state["eval_lb"]
        trainer.ensemble = state["ensemble"]
        trainer.best_model = state["best_model"]

        trainer.X_train_full = state["X_train_full"]
        trainer.y_train_full = state["y_train_full"]
        trainer.X_eval = state["X_eval"]
        trainer.y_eval = state["y_eval"]

        trainer.meta_study = state["meta_study"]
        trainer.meta_dict = state["meta_dict"]
        trainer.meta_learner = state["meta_learner"]
        trainer.meta_scaler = state["meta_scaler"]

        print(f"Trainer successfully loaded from {path}.")
        return trainer