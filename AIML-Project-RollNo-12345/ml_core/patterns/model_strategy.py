from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class ModelStrategy(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, model, X_test):
        pass

class LogisticRegressionStrategy(ModelStrategy):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if 'max_iter' not in self.kwargs:
            self.kwargs['max_iter'] = 1000

    def train(self, X_train, y_train):
        model = LogisticRegression(**self.kwargs)
        model.fit(X_train, y_train)
        return model
    
    def predict(self, model, X_test):
        return model.predict(X_test)
    
    def predict_proba(self, model, X_test):
        return model.predict_proba(X_test)

class RandomForestStrategy(ModelStrategy):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def train(self, X_train, y_train):
        model = RandomForestClassifier(**self.kwargs)
        model.fit(X_train, y_train)
        return model
    
    def predict(self, model, X_test):
        return model.predict(X_test)
    
    def predict_proba(self, model, X_test):
        return model.predict_proba(X_test)
