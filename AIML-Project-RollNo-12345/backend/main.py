from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import StudentProfile, PredictionResponse
import pandas as pd
import sys
import os
import shap

# Add ml_core to path so we can import patterns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml_core')))
from patterns.model_registry import ModelRegistry
from patterns.data_factory import DataProcessorFactory
from patterns.model_strategy import RandomForestStrategy

app = FastAPI(title="Student Placement Prediction API")

# Configure CORS for Vite Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Singleton on Startup
@app.on_event("startup")
def load_model():
    registry = ModelRegistry()
    model_data = registry.get_model()
    if not model_data:
        print("Warning: Model not found. Please run ml_core/train.py first.")

@app.post("/predict", response_model=PredictionResponse)
def predict_placement(profile: StudentProfile):
    registry = ModelRegistry()
    model_data = registry.get_model()
    if not model_data:
        raise HTTPException(status_code=500, detail="Model is not trained yet.")
        
    model = model_data['model']
    expected_columns = model_data['columns']
    
    # Create DataFrame from profile
    df = pd.DataFrame([profile.dict()])
    
    # Process data
    processor = DataProcessorFactory.get_processor("standard")
    processor.expected_columns = expected_columns
    X, _ = processor.preprocess(df, is_training=False)
    
    # Predict using Strategy
    strategy = RandomForestStrategy()
    prob = strategy.predict_proba(model, X)[0][1]
    pred = strategy.predict(model, X)[0]
    
    # Generate dynamic recommendations based on input
    recommendations = []
    
    # Feature Importance (SHAP)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # In newer SHAP versions for binary classification, shap_values might be a list or array
    if isinstance(shap_values, list):
        # Index 1 is the positive class (Placed)
        sv = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]
        
    feature_impact = {}
    for col, val in zip(X.columns, sv):
        # Store impact formatted to 4 decimals
        feature_impact[col] = round(float(val), 4)

    # Use SHAP values to generate dynamic recommendations
    # Find the feature that negatively impacted the score the most
    if len(feature_impact) > 0:
        worst_feature = min(feature_impact, key=feature_impact.get)
        if worst_feature == 'workex_Yes':
            recommendations.append("Prioritize getting an internship or part-time work experience; this heavily influences placement.")
        elif worst_feature == 'mba_p':
            recommendations.append("Your MBA percentage is pulling your score down. Focus intensely on core MBA subjects and case studies.")
        elif worst_feature == 'degree_p':
            recommendations.append("Your undergraduate degree score is a weak point. Highlight specific strong projects in your resume.")
        elif worst_feature == 'etest_p':
            recommendations.append("Your employability test score suggests you should practice aptitude and technical mock tests.")
        elif worst_feature == 'academic_average':
            recommendations.append("Your overall academic average is below expectations. Focus heavily on practical skills and certifications.")
        else:
            recommendations.append(f"Consider improving your profile related to {worst_feature}.")

    if not recommendations:
        recommendations.append("Your profile is strong! Focus on mock interviews and resume polishing.")
        
    return PredictionResponse(
        placement_probability=round(prob * 100, 2),
        prediction="Placed" if pred == 1 else "Not Placed",
        recommendations=recommendations,
        feature_importance=feature_impact
    )

@app.get("/analytics")
def get_analytics():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Placement_Data_Full_Class.csv')
    if not os.path.exists(data_path):
        return {"error": "Dataset not found"}
        
    df = pd.read_csv(data_path)
    
    # Calculate some basic aggregations
    specialisation_placement = df.groupby(['specialisation', 'status']).size().unstack(fill_value=0).to_dict(orient="index")
    workex_placement = df.groupby(['workex', 'status']).size().unstack(fill_value=0).to_dict(orient="index")
    
    return {
        "specialisation_stats": specialisation_placement,
        "workex_stats": workex_placement
    }
