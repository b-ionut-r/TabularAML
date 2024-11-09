DEFAULT_CONFIG = {
                     
                    "XGB": {
                        "learning_rate": (0.005, 0.3, "float_log"), 
                        "n_estimators": (50, 1000, "int"),
                        "max_depth": (3, 12, "int"),
                        "min_child_weight": (1, 10, "int"),
                        "subsample": (0.5, 1.0, "float"),
                        "colsample_bytree": (0.5, 1.0, "float"),
                        "reg_lambda": (1e-3, 10, "float_log"), 
                        "reg_alpha": (1e-3, 5, "float_log"),  
                        "gamma": (0.0, 1.0, "float"), 
                        "scale_pos_weight": (1e-3, 50.0, "float_log"),
                        "verbosity": 0,
                        "priority": 120
                    },
                    
                    "LGB": {
                        "metric": (["None"], "cat"),
                        "learning_rate": (0.005, 0.2, "float_log"),  
                        "n_estimators": (50, 1000, "int"),
                        "max_depth": (3, 20, "int"),
                        "num_leaves": (20, 255, "int"),
                        "feature_fraction": (0.5, 1.0, "float"),
                        "bagging_fraction": (0.5, 1.0, "float"),
                        "lambda_l1": (1e-3, 10.0, "float_log"), 
                        "lambda_l2": (1e-3, 10.0, "float_log"),  
                        "min_child_samples": (5, 100, "int"),
                        "boost_from_average": ([True, False], "cat"),
                        "verbosity": -1,
                        "priority": 110
                    },
                    
                    "CAT": {
                        "learning_rate": (0.005, 0.2, "float_log"),  
                        "n_estimators": (50, 1000, "int"),
                        "max_depth": (4, 10, "int"),
                        "reg_lambda": (1e-3, 10.0, "float_log"), 
                        "random_strength": (1e-3, 10.0, "float_log"),  
                        "bagging_temperature": (0.0, 1.0, "float"),
                        "border_count": (32, 255, "int"),
                        "min_data_in_leaf": (1, 50, "int"), 
                        "verbose": False,
                        "priority": 100
                    },

                    "RF": {
                        "n_estimators": (50, 500, "int"),
                        "max_depth": (5, 50, "int"),
                        "min_samples_split": (2, 20, "int"),
                        "min_samples_leaf": (1, 20, "int"),
                        "max_features": (["sqrt", "log2", None], "cat"),
                        "priority": 70
                    },


                    "SGD_LINEAR": {
                        "alpha": (1e-6, 1e-1, "float_log"),
                        "penalty": (["l2", "l1", "elasticnet"], "cat"),
                        "l1_ratio": (0.1, 0.9, "float"),
                        "learning_rate": (["constant", "optimal", "invscaling", "adaptive"], "cat"),
                        "eta0": (1e-4, 1e-1, "float_log"),
                        "power_t": (0.1, 0.5, "float"),
                        "max_iter": (500, 3000, "int"),
                        "tol": (1e-5, 1e-3, "float_log"),
                        "shuffle": ([True, False], "cat"),
                        "early_stopping": ([True, False], "cat"),
                        "n_iter_no_change": (3, 10, "int"),
                        "priority": 50
                    }

                }    