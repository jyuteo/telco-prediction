import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from enum import Enum
from utils import reset_seed
from scipy.stats import randint, loguniform


class ResamplingClassifier(BaseEstimator):
    def __init__(self, model, resampler):
        self.resampler = resampler
        self.model = model

    def fit(self, X, y, **fit_params):
        X, y = self.resampler.fit_resample(X, y)
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_log_proba(self, X):
        return self.model.predict_log_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)


class ModelGenerator:
    class MODEL(Enum):
        LR = 0
        GBC = 1
        RF = 2
        STACK = 3
        MLP = 4

    class RESAMPLER(Enum):
        OVERSAMPLING = 0
        UNDERSAMPLING = 1
        SMOTE = 2

    def __init__(self):
        self.__distribution = {
            self.MODEL.LR: {
                "C": loguniform(0.001, 10),
                "class_weight": [{
                    0: i,
                    1: 1.
                } for i in np.logspace(-1, 0, 100)]
            },
            self.MODEL.GBC: {
                "n_estimators": randint(1, 1001),
                "learning_rate": loguniform(0.0001, 1),
                "max_depth": randint(3, 6),
                "min_samples_split": [25, 30, 35, 40],
                "min_samples_leaf": [5, 10, 15, 20],
            },
            self.MODEL.RF: {
                "n_estimators": randint(1, 1001),
                "max_depth": randint(3, 6),
                "min_samples_split": [25, 30, 35, 40],
                "min_samples_leaf": [5, 10, 15, 20],
                "class_weight": [{
                    0: i,
                    1: 1.
                } for i in np.logspace(-1, 0, 100)],
            },
            self.MODEL.STACK: {
                "gb__n_estimators":
                randint(1, 1001),
                "gb__learning_rate":
                loguniform(0.0001, 10),
                "gb__max_depth":
                randint(3, 6),
                "gb__min_samples_split": [25, 30, 35, 40],
                "gb__min_samples_leaf": [5, 10, 15, 20],
                "lr__C":
                loguniform(0.001, 10),
                "lr__class_weight": [{
                    0: i,
                    1: 1.
                } for i in np.logspace(-1, 0, 100)],
                "rf__n_estimators":
                randint(1, 1001),
                "rf__max_depth":
                randint(3, 6),
                "rf__min_samples_split": [25, 30, 35, 40],
                "rf__min_samples_leaf": [5, 10, 15, 20],
                "rf__class_weight": [{
                    0: i,
                    1: 1.
                } for i in np.logspace(-1, 0, 100)],
                "final_estimator__C":
                loguniform(0.001, 10),
                "final_estimator__class_weight": [{
                    0: i,
                    1: 1.
                } for i in np.logspace(-1, 0, 100)],
            },
            self.MODEL.MLP: {
                "hidden_layer_sizes": [(i, j, k) for i in range(50, 101)
                                       for j in range(25, 51)
                                       for k in range(10, 26)],
                "learning_rate_init":
                loguniform(0.00001, 1),
                "alpha":
                loguniform(0.001, 10),
            }
        }

    def _get_model(self, name):
        reset_seed()
        if name == self.MODEL.LR:
            return LogisticRegression(max_iter=5000, random_state=0)
        elif name == self.MODEL.GBC:
            return GradientBoostingClassifier(random_state=0)
        elif name == self.MODEL.RF:
            return RandomForestClassifier(random_state=0)
        elif name == self.MODEL.STACK:
            sc = StackingClassifier(
                estimators=[("gb", GradientBoostingClassifier(random_state=0)),
                            ("lr",
                             LogisticRegression(max_iter=5000,
                                                random_state=0)),
                            ("rf",
                             RandomForestClassifier(min_samples_split=20,
                                                    min_samples_leaf=10,
                                                    random_state=0))],
                final_estimator=LogisticRegression(class_weight={
                    0: 0.6,
                    1: 1.
                },
                                                   random_state=0))
            return sc
        elif name == self.MODEL.MLP:
            return MLPClassifier(max_iter=5000,
                                 early_stopping=True,
                                 validation_fraction=0.2,
                                 n_iter_no_change=50,
                                 learning_rate="adaptive",
                                 batch_size=1000,
                                 random_state=0)
        raise Exception("invalid model {}".format(name))

    def _get_resampler(self, name):
        reset_seed()
        if name is None:
            return None
        elif name == self.RESAMPLER.OVERSAMPLING:
            return RandomOverSampler(random_state=0)
        elif name == self.RESAMPLER.UNDERSAMPLING:
            return RandomUnderSampler(random_state=0)
        elif name == self.RESAMPLER.SMOTE:
            return SMOTE(random_state=0)
        raise Exception("invalid resampler {}".format(name))

    def _get_distribution(self, name):
        if name in self.__distribution:
            return self.__distribution[name]
        raise Exception("invalid model {}".format(name))

    @staticmethod
    def _build_pipeline(model,
                        numerical_features,
                        categorical_features,
                        resampler=None,
                        param_distributions=None):
        reset_seed()
        col_transform = ColumnTransformer([
            ("cat", OneHotEncoder(), categorical_features),
            ("num", StandardScaler(), numerical_features)
        ])
        if resampler is not None:
            model = ResamplingClassifier(model, resampler)
        pipe = Pipeline([("col_transform", col_transform), ("model", model)])

        if isinstance(param_distributions, dict):
            dist = dict()
            for k, v in param_distributions.items():
                if resampler is not None:
                    k = "model__model__{}".format(k)
                else:
                    k = "model__{}".format(k)
                dist[k] = v

            pipe = RandomizedSearchCV(estimator=pipe,
                                      param_distributions=dist,
                                      verbose=10,
                                      n_iter=50,
                                      scoring="f1",
                                      refit=False,
                                      cv=5,
                                      n_jobs=5,
                                      random_state=0)
        return pipe

    def generate_models(self,
                        numerical_features,
                        categorical_features,
                        hyperparameter_tuning=True):
        models = {
            "logistic-regression": {
                "model": self.MODEL.LR,
            },
            "logistic-regression-oversampling": {
                "model": self.MODEL.LR,
                "resampler": self.RESAMPLER.OVERSAMPLING,
            },
            "logistic-regression-undersampling": {
                "model": self.MODEL.LR,
                "resampler": self.RESAMPLER.UNDERSAMPLING,
            },
            "logistic-regression-smote": {
                "model": self.MODEL.LR,
                "resampler": self.RESAMPLER.SMOTE,
            },
            "gradient-boosting": {
                "model": self.MODEL.GBC,
            },
            "gradient-boosting-oversampling": {
                "model": self.MODEL.GBC,
                "resampler": self.RESAMPLER.OVERSAMPLING,
            },
            "gradient-boosting-undersampling": {
                "model": self.MODEL.GBC,
                "resampler": self.RESAMPLER.UNDERSAMPLING,
            },
            "gradient-boosting-smote": {
                "model": self.MODEL.GBC,
                "resampler": self.RESAMPLER.SMOTE,
            },
            "random-forest": {
                "model": self.MODEL.RF,
            },
            "random-forest-oversampling": {
                "model": self.MODEL.RF,
                "resampler": self.RESAMPLER.OVERSAMPLING,
            },
            "random-forest-undersampling": {
                "model": self.MODEL.RF,
                "resampler": self.RESAMPLER.UNDERSAMPLING,
            },
            "random-forest-smote": {
                "model": self.MODEL.RF,
                "resampler": self.RESAMPLER.SMOTE,
            },
            "stacking-gbs-lr-rf": {
                "model": self.MODEL.STACK,
            },
            "mlp-oversampling": {
                "model": self.MODEL.MLP,
                "resampler": self.RESAMPLER.OVERSAMPLING,
            },
            "mlp-smote": {
                "model": self.MODEL.MLP,
                "resampler": self.RESAMPLER.SMOTE,
            },
        }

        for model_name, kwargs in models.items():
            yield model_name, self._build_pipeline(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                model=self._get_model(kwargs["model"]),
                resampler=self._get_resampler(kwargs.get("resampler", None)),
                param_distributions=self._get_distribution(kwargs["model"])
                if hyperparameter_tuning else None)
