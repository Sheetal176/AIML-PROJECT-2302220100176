import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from patterns.data_factory import DataProcessorFactory
from patterns.model_registry import ModelRegistry
from sklearn.metrics import accuracy_score

def train_pipeline():
    print("Loading dataset...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Placement_Data_Full_Class.csv')
    df = pd.read_csv(data_path)
    
    print("Processing data...")
    processor = DataProcessorFactory.get_processor("standard")
    X, y = processor.preprocess(df, is_training=True)
    
    # Save the expected columns for the processor
    expected_columns = list(X.columns)
    processor.expected_columns = expected_columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Running GridSearchCV for Random Forest...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    
    # Evaluate
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Optimized Model Accuracy: {acc:.4f}")
    
    print("Saving optimized model via registry...")
    registry = ModelRegistry()
    
    # Save model and columns
    registry.save_model({
        'model': best_model,
        'columns': expected_columns
    })
    print("Pipeline complete.")

if __name__ == "__main__":
    train_pipeline()

