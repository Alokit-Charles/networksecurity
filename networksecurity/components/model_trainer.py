import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.constants.training_pipeline import TARGET_COLUMN


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

import mlflow


class ModelTrainer:
    def __init__(self, data_transformation_artifact : DataTransformationArtifact, model_trainer_config : ModelTrainerConfig):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classification_metric : ClassificationMetricArtifact):
        import dagshub
        dagshub.init(repo_owner='Alokit-Charles', repo_name='networksecurity', mlflow=True)

        f1_score = classification_metric.f1_score
        precision_score = classification_metric.precision_score
        recall_score = classification_metric.recall_score

        mlflow.set_tracking_uri("https://dagshub.com/Alokit-Charles/networksecurity.mlflow")
        mlflow.set_experiment("network-security")
        
        with mlflow.start_run(run_name="networksecurity_model"):
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision score", precision_score)
            mlflow.log_metric("recall score", recall_score)
            mlflow.sklearn.log_model(sk_model= best_model, name = 'model')

        logging.info("Experimentation Metrics saved in mlflow.")

    
    def train_model(self, X_train, y_train, X_test, y_test):
        models  = {
            "Logistic Regression" : LogisticRegression(verbose= 1),
            "KNeighbors" : KNeighborsClassifier(),
            "Decision Tree" : DecisionTreeClassifier(),
            "Random Forest" : RandomForestClassifier(verbose= 1),
            "Gradient Boosting" : GradientBoostingClassifier(verbose= 1),
            "AdaBoost" : AdaBoostClassifier()
        }

        params = {
            
                "Logistic Regression": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l2"],
                # "solver": ["lbfgs", "liblinear"]
            },

            "KNeighbors": {
                "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                # "leaf_size": [20, 30, 40, 50],
                # "p": [1, 2]
            },

            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
                "splitter": ["best", "random"],
                "max_depth": [None, 5, 10, 20, 30, 50],
                # "min_samples_split": [2, 5, 10, 20],
                # "min_samples_leaf": [1, 2, 4, 8],
                "max_features": [None, "sqrt", "log2"]
            },

            "Random Forest": {
                "n_estimators": [100, 200, 300, 500],
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [None, 10, 20, 30, 50],
                # "min_samples_split": [2, 5, 10],
                # "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
                # "bootstrap": [True, False]
            },

            "Gradient Boosting": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "max_depth": [3, 5, 7],
                # "min_samples_split": [2, 5, 10],
                # "min_samples_leaf": [1, 2, 4]
            },

            "AdaBoost": {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
            }
        
        }

        logging.info("Evaluating best Model.")
        
        model_report: dict = evaluate_models(X_train= X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                              models = models, params = params)

        ## get best model score
        best_model_score = max(sorted(model_report.values()))
        ## get best model name
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

        logging.info(f"The best model is : {best_model_name} with accuracy score : {best_model_score}")

        best_model = models[best_model_name]
        y_train_pred = best_model.predict(X_train)

        classification_train_metric = get_classification_score(y_true= y_train, y_pred= y_train_pred)


        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true= y_test, y_pred= y_test_pred)

        ## track mlflow
        
        self.track_mlflow(best_model, classification_train_metric)
        self.track_mlflow(best_model, classification_test_metric)

        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)


        Network_Model = NetworkModel(preprocessor= preprocessor, model = best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj = Network_Model) 

        ## model pusher
        save_object(self.model_trainer_config.finalized_model_object_path, Network_Model)

        ## Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path = self.model_trainer_config.trained_model_file_path, 
                             train_metric_artifact= classification_train_metric,
                             test_metric_artifact= classification_test_metric)

        logging.info(f"Model trainer artifact created.")

        return model_trainer_artifact


    def inititate_model_trainer(self) -> ModelTrainerArtifact:

        logging.info("Entered initiate_model_trainer of ModelTrainer class.")
        try:
            self.train_file_path = self.data_transformation_artifact.transformed_train_file_path
            self.test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(self.train_file_path)
            test_arr = load_numpy_array_data(self.test_file_path)

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            model = self.train_model(X_train, y_train, X_test, y_test)

            return model

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        