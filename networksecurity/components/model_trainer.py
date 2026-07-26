import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.constants.training_pipeline import TARGET_COLUMN


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier



class ModelTrainer:
    def __init__(self, model_trainer_config : ModelTrainerConfig, data_transformation_artifact : DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

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
            "Decision Tree" : {
                'criterion' : ['gini', 'entroy', 'log_loss']
            },
            "Random Forest" : {
                'n_estimators' : [8, 16, 32, 64, 128]
            },
            "Gradient Boosting" : {
                'learning_rate' : [0.1, 0.01, 0.05, 0.01],
                'subsample' : [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
            },
            "Logistic Regression" : {},
            "AdaBoost" : {
                'learning_rate' : [0.1, 0.01, 0.5, 0.001],
                'n_estimators' : [8, 16, 32, 64, 128, 256]
            },
            "KNeighbors" : {
                'n_neighbors' : [3, 5, 10, 12, 7],
                'algorithm' : ['ball_tree', 'kd_tree']
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

        ## track mlflow

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true= y_test, y_pred= y_test_pred)

        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = self.model_trainer_config.trained_model_file_path

        # logging.info(f"model dir path : {model_dir_path}")
        # os.makedirs(model_dir_path, exist_ok= True)

        Network_Model = NetworkModel(preprocessor= preprocessor, model = best_model)
        save_object(model_dir_path, obj = Network_Model) 

        ## Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path = model_dir_path, 
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
        