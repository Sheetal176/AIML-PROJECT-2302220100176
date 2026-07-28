import os
import pickle

class ModelRegistry:
    _instance = None
    _model = None
    _model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        if os.path.exists(self._model_path):
            with open(self._model_path, 'rb') as f:
                self._model = pickle.load(f)
        else:
            print(f"Warning: Model not found at {self._model_path}")
            
    def get_model(self):
        return self._model
    
    def save_model(self, model):
        with open(self._model_path, 'wb') as f:
            pickle.dump(model, f)
        self._model = model
