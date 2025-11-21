import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import json

# Config
API_URL = "http://localhost:8000"

st.set_page_config(page_title="FraudLM: Explainable Fraud Detection", layout="wide")

# --- Sidebar: Business Simulator ---
st.sidebar.title("💰 Business Impact Simulator")
st.sidebar.markdown("Adjust costs to estimate savings.")

vol_monthly = st.sidebar.number_input("Monthly Transactions", value=1_000_000)
fraud_rate = st.sidebar.slider("Fraud Rate (%)", 0.0, 5.0, 0.5) / 100
avg_fraud_loss = st.sidebar.number_input("Avg Fraud Loss ($)", value=200)
cost_false_pos = st.sidebar.number_input("Cost per False Positive ($)", value=5) # Support cost + churn

# Simple Impact Calc
st.sidebar.markdown("---")
st.sidebar.subheader("Projected Savings")
# Hypothetical improvements
baseline_accuracy = 0.90 # Rule based
model_accuracy = 0.96 # ML based

savings = (vol_monthly * fraud_rate * avg_fraud_loss) * (model_accuracy - baseline_accuracy)
st.sidebar.metric("Monthly Fraud Savings", f"${savings:,.0f}")
st.sidebar.metric("Annual Savings", f"${savings * 12:,.0f}")


# --- Main Content ---
st.title("🛡️ FraudLM: Explainable AI for Fraud Detection")
st.markdown("Real-time fraud scoring with LLM-driven explanations for analysts and customers.")

# Tabs
tab1, tab2 = st.tabs(["🛑 Live Transaction Monitor", "📊 Model Performance"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Simulate Transaction")
        
        # Manual Input or Random
        with st.expander("Transaction Details", expanded=True):
            amt = st.number_input("Amount ($)", value=150.0)
            time_diff = st.slider("Hours since last tx", 0.0, 48.0, 1.5)
            amt_ratio = st.slider("Amount vs User Avg (Ratio)", 0.1, 20.0, 3.5)
            country_change = st.checkbox("Country Changed?", value=True)
            merch_freq = st.slider("Merchant Category Frequency", 0.0, 1.0, 0.1)
            
            payload = {
                "amount": amt,
                "time_diff": time_diff,
                "amount_vs_avg": amt_ratio,
                "country_change": 1 if country_change else 0,
                "merchant_cat_freq": merch_freq,
                "user_id": "user_123",
                "country": "FR" if country_change else "US"
            }
            
        if st.button("Analyze Transaction", type="primary"):
            try:
                # 1. Get Prediction
                with st.spinner("Running XGBoost Model..."):
                    pred_res = requests.post(f"{API_URL}/predict", json=payload).json()
                
                # 2. Get Explanation
                with st.spinner("Generating LLM Explanation..."):
                    explain_payload = {
                        "transaction": payload,
                        "probability": pred_res['probability'],
                        "decision": pred_res['decision'],
                        "top_risk_factors": pred_res['top_risk_factors'],
                        "audience": "analyst"
                    }
                    analyst_expl = requests.post(f"{API_URL}/explain", json=explain_payload).json()
                    
                    explain_payload['audience'] = 'customer'
                    cust_expl = requests.post(f"{API_URL}/explain", json=explain_payload).json()

                # Store in session state to persist
                st.session_state['result'] = pred_res
                st.session_state['analyst_expl'] = analyst_expl['explanation']
                st.session_state['cust_expl'] = cust_expl['explanation']
                
            except Exception as e:
                st.error(f"Error connecting to API: {e}")

    with col2:
        if 'result' in st.session_state:
            res = st.session_state['result']
            
            # Status Badge
            if res['decision'] == "FLAGGED":
                st.error(f"🚨 FLAGGED (Score: {res['probability']:.1%})")
            else:
                st.success(f"✅ APPROVED (Score: {res['probability']:.1%})")
            
            # Dual View Toggle
            view_mode = st.radio("View Mode", ["👮 Analyst View", "🤝 Customer View"], horizontal=True)
            
            if view_mode == "👮 Analyst View":
                st.markdown("### LLM Assessment")
                st.info(st.session_state['analyst_expl'])
                
                st.markdown("### Risk Drivers (SHAP)")
                # Simple Bar Chart for SHAP
                factors = res['top_risk_factors']
                df_shap = pd.DataFrame(factors)
                if not df_shap.empty:
                    # Color by increase/decrease
                    df_shap['color'] = df_shap['shap_value'].apply(lambda x: 'red' if x > 0 else 'green')
                    st.bar_chart(df_shap.set_index('feature')['shap_value'])
                
            else:
                st.markdown("### Customer Notification Draft")
                st.success(st.session_state['cust_expl'])
                st.caption("This message is ready to be sent via SMS/Email.")

with tab2:
    st.markdown("### Model Metrics")

    metrics_path = "src/model/artifacts/metrics.json"

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        st.success("Loaded model metrics from training artifacts.")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("ROC AUC", f"{metrics.get('roc_auc', 0.0):.3f}")
            st.metric("PR AUC", f"{metrics.get('pr_auc', 0.0):.3f}")
            st.markdown("#### Classification Report")
            st.code(metrics.get("class_report", ""), language="text")

        with col2:
            roc = metrics.get("roc_curve", {})
            fpr = roc.get("fpr", [])
            tpr = roc.get("tpr", [])

            if fpr and tpr:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random",
                        line=dict(dash="dash"),
                    )
                )
                fig.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                )
                st.plotly_chart(fig, use_container_width=True)

            cm = metrics.get("confusion_matrix")
            if cm:
                st.markdown("#### Confusion Matrix")
                cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
                st.dataframe(cm_df)

    except FileNotFoundError:
        st.warning("Metrics artifacts not found. Run `python src/model/train.py` to generate them.")
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
