import os
import joblib
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Fraud Detection & Explainability API")

# Load Artifacts (Global scope to load once)
print("Loading model artifacts...")
try:
    model = joblib.load('src/model/artifacts/model.joblib')
    shap_explainer = joblib.load('src/model/artifacts/shap_explainer.joblib')
    feature_names = joblib.load('src/model/artifacts/features.joblib')
except FileNotFoundError:
    print("WARNING: Model artifacts not found. Run src/model/train.py first.")
    model = None
    shap_explainer = None
    feature_names = []

# Initialize LLM Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Pydantic Models ---
class TransactionFeatures(BaseModel):
    amount: float
    time_diff: float
    amount_vs_avg: float
    country_change: int
    merchant_cat_freq: float
    
    # Optional metadata for the LLM prompt (not for the model)
    user_id: str = "unknown"
    merchant: str = "unknown"
    country: str = "unknown"

class PredictionResponse(BaseModel):
    probability: float
    is_fraud: bool
    decision: str
    shap_values: Dict[str, float]
    top_risk_factors: List[Dict[str, Any]]

class ExplanationRequest(BaseModel):
    transaction: Dict[str, Any]
    probability: float
    decision: str
    top_risk_factors: List[Dict[str, Any]]
    audience: str = "analyst" # or "customer"

class ExplanationResponse(BaseModel):
    explanation: str

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(features: TransactionFeatures):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Prepare input dataframe in correct order
    input_data = pd.DataFrame([features.dict()])
    input_features = input_data[feature_names]
    
    # Predict
    prob = float(model.predict_proba(input_features)[0][1])
    threshold = 0.4 # Tuned threshold (hypothetical)
    is_fraud = prob > threshold
    decision = "FLAGGED" if is_fraud else "APPROVED"
    
    # SHAP Values
    # TreeExplainer returns a numpy array for interactions, we want raw values
    shap_values = shap_explainer.shap_values(input_features)
    
    # If binary classification, shap_values might be a list [class0, class1] or just class1
    # XGBoost + TreeExplainer usually returns just the margin/log-odds for the raw output
    # But let's handle the array shape carefully.
    if isinstance(shap_values, list):
        sv = shap_values[1][0] # Positive class
    else:
        sv = shap_values[0]
        
    # Map features to SHAP values
    feature_shap = dict(zip(feature_names, sv))
    
    # Identify Top Risk Factors (High positive SHAP pushes towards Fraud)
    # We sort by magnitude but only care about what pushes risk UP for the explanation
    risk_factors = []
    for feat, val in feature_shap.items():
        risk_factors.append({
            "feature": feat,
            "shap_value": float(val),
            "value": float(input_features.iloc[0][feat]),
            "impact": "Increase Risk" if val > 0 else "Decrease Risk"
        })
    
    # Sort by absolute SHAP importance
    risk_factors.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    
    return {
        "probability": prob,
        "is_fraud": is_fraud,
        "decision": decision,
        "shap_values": feature_shap,
        "top_risk_factors": risk_factors[:3] # Top 3 drivers
    }

@app.post("/explain", response_model=ExplanationResponse)
def generate_explanation(req: ExplanationRequest):
    """
    Generates a natural language explanation using the LLM.
    """
    
    if req.audience == "analyst":
        system_prompt = """You are an expert Fraud Analyst AI assistant. 
        Explain the fraud model's decision for the technical team.
        Focus on the SHAP key drivers. Be precise, professional, and concise."""
        
        user_prompt = f"""
        Transaction Context: {req.transaction}
        Model Decision: {req.decision} (Score: {req.probability:.2f})
        
        Key Risk Factors (SHAP):
        {req.top_risk_factors}
        
        Explain why this was flagged.
        """
        
    else: # Customer
        system_prompt = """You are a helpful Customer Support Agent for a bank.
        Explain to a customer why their transaction was flagged/blocked in simple, reassuring terms.
        Do NOT use technical jargon like 'SHAP' or 'Model Score'.
        Focus on unusual activity patterns (location, amount, frequency).
        """
        
        user_prompt = f"""
        Transaction Context: {req.transaction}
        Key Factors:
        {req.top_risk_factors}
        
        Write a short message to the user."""

    try:
        response = client.chat.completions.create(
            model="gpt-5.1", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=600
        )

        explanation = None

        # Primary: standard chat.completions style
        try:
            if response and getattr(response, "choices", None):
                explanation = response.choices[0].message.content
        except Exception:
            explanation = None

        # Fallback: responses-style objects (output_text / output)
        if not explanation:
            if hasattr(response, "output_text") and response.output_text:
                explanation = response.output_text
            elif hasattr(response, "output") and response.output:
                parts = []
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for content in item.content:
                            text_val = getattr(content, "text", None)
                            if text_val:
                                parts.append(text_val)
                if parts:
                    explanation = "".join(parts)

        if not explanation:
            explanation = "LLM returned empty content."

    except Exception as e:
        explanation = f"Error generating explanation: {str(e)}"
        
    return {"explanation": explanation}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
