import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants ---
ARTIFACTS_DIR = "./code/thomas/model/model_artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "food_security_model.pth")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.save")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.save")
HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "training_history.json")
CLUSTERING_PATH = os.path.join(ARTIFACTS_DIR, "clustering_model.save")
CLUSTERED_DATA_PATH = os.path.join(ARTIFACTS_DIR, "data_with_clusters.csv")

# --- Model Class Definition (Must match training script exactly) ---
class FoodSecurityFFNN(nn.Module):
    def __init__(self, input_dim):
        super(FoodSecurityFFNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)

        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.act3 = nn.ReLU()
        
        self.layer4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.layer1(x))))
        x = self.drop2(self.act2(self.bn2(self.layer2(x))))
        x = self.act3(self.bn3(self.layer3(x)))
        return self.layer4(x)

# --- Helper Functions ---

@st.cache_resource
def load_resources():
    """Load model, scaler, metadata, clustering data once to optimize speed."""
    if not os.path.exists(MODEL_PATH):
        st.write(os.getcwd())
        st.error("Model files not found! Please run 'train_and_save_model.py' first.")
        return None, None, None, None, None, None

    # Load Metadata
    metadata = joblib.load(METADATA_PATH)

    # Load Scaler
    scaler = joblib.load(SCALER_PATH)

    # Load Model
    input_dim = len(metadata['feature_columns'])
    model = FoodSecurityFFNN(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    # Load History
    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)

    # Load Clustering Data
    clustering_data = None
    clustered_df = None
    if os.path.exists(CLUSTERING_PATH):
        clustering_data = joblib.load(CLUSTERING_PATH)
    if os.path.exists(CLUSTERED_DATA_PATH):
        clustered_df = pd.read_csv(CLUSTERED_DATA_PATH)

    return model, scaler, metadata, history, clustering_data, clustered_df

def make_prediction(model, scaler, metadata, input_data):
    """Preprocesses input and runs inference."""
    # 1. Convert dict to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # 2. One-Hot Encoding
    # We must ensure categorical columns are strings just like in training
    for col in metadata['categorical_cols']:
        df_input[col] = df_input[col].astype(str)
        
    df_encoded = pd.get_dummies(df_input, columns=metadata['categorical_cols'], drop_first=True)
    
    # 3. Reindex to match Training Columns
    # This is CRITICAL. It adds missing columns (with 0) and removes extra ones.
    df_encoded = df_encoded.reindex(columns=metadata['feature_columns'], fill_value=0)
    
    # 4. Split into Numeric and Categorical parts
    X_numeric = df_encoded[metadata['numeric_cols']].values
    X_categorical = df_encoded[metadata['encoded_cat_cols']].values.astype(float)
    
    # 5. Scale Numeric
    X_numeric_scaled = scaler.transform(X_numeric)
    
    # 6. Combine
    X_final = np.hstack([X_numeric_scaled, X_categorical])
    
    # 7. Predict
    tensor_input = torch.FloatTensor(X_final)
    with torch.no_grad():
        prediction = model(tensor_input).item()
        
    return prediction

# --- Streamlit Layout ---

def nn_webpage():
    model, scaler, metadata, history, clustering_data, clustered_df = load_resources()

    if model is not None:

        st.subheader("Enter County Economic Data")
        
        col1, col2, col3 = st.columns(3)
        
        input_data = {}
        
        # Numeric Inputs
        with col1:
            st.markdown("### 💰 Income & Poverty")
            input_data['MEDIAN_HOUSEHOLD_INCOME'] = st.number_input("Median Household Income ($)", min_value=0, value=50000)
            input_data['POVERTY_RATE'] = st.number_input("Poverty Rate (%)", min_value=0.0, max_value=100.0, value=15.0)
            
        with col2:
            st.markdown("### 🏗️ Employment & Aid")
            input_data['UNEMPLOYMENT_RATE'] = st.number_input("Unemployment Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
            input_data['SNAP_RECEIPT_RATE'] = st.number_input("SNAP Receipt Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
            
        with col3:
            st.markdown("### 👥 Population Context")
            input_data['POP_16_PLUS'] = st.number_input("Population (16+)", min_value=0, value=10000)
            input_data['HOUSEHOLDS_TOTAL'] = st.number_input("Total Households", min_value=0, value=5000)

        st.markdown("---")
        st.markdown("### 📍 Categorical Context")
        cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)

        # Categorical Dropdowns (Populated from Metadata)
        with cat_col1:
            input_data['State'] = st.selectbox("State", metadata['unique_values']['State'])
        with cat_col2:
            input_data['YEAR'] = st.selectbox("Year (Model Context)", metadata['unique_values']['YEAR'], index=len(metadata['unique_values']['YEAR'])-1)
        with cat_col3:
            input_data['Low_Threshold_Type'] = st.selectbox("Threshold Type", metadata['unique_values']['Low_Threshold_Type'])
        with cat_col4:
            if 'Cluster' in metadata['unique_values']:
                input_data['Cluster'] = st.selectbox("Cluster Group", metadata['unique_values']['Cluster'])
            else:
                input_data['Cluster'] = st.selectbox("Cluster Group", ['0', '1'], help="Cluster assignment based on poverty/employment features")

        # Prediction Button
        if st.button("Predict Food Insecurity Rate", type="primary"):
            pred = make_prediction(model, scaler, metadata, input_data)
            
            st.divider()
            st.success(f"### Predicted Rate: {pred:.2f}%")
            
            # Visual Gauge
            st.progress(min(pred / 50.0, 1.0)) # Assumes 50% is a very high extreme for visual scaling
            if pred < 10:
                st.caption("This indicates a relatively low level of food insecurity.")
            elif pred < 20:
                st.caption("This indicates a moderate level of food insecurity.")
            else:
                st.warning("This indicates a high level of food insecurity.")
        
        st.subheader("Training History")
        
        # Plotly Interactive Graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history['train_loss'], mode='lines', name='Training Loss'))
        fig.add_trace(go.Scatter(y=history['val_loss'], mode='lines', name='Validation Loss'))
        
        fig.update_layout(
            title='Loss Curve (MSE) Over Epochs',
            xaxis_title='Epoch',
            yaxis_title='Mean Squared Error',
            template='plotly_white',
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("Lower loss indicates better model performance. The gap between Training and Validation shows how well the model generalizes.")

        # --- Model Performance Metrics ---
        st.markdown("---")
        st.header("📈 Model Performance Metrics")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.subheader("Training Convergence")
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            improvement = ((history['train_loss'][0] - final_train_loss) / history['train_loss'][0]) * 100

            st.metric("Final Training Loss", f"{final_train_loss:.4f}")
            st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
            st.metric("Improvement from Start", f"{improvement:.1f}%")

            # Overfitting indicator
            gap = abs(final_train_loss - final_val_loss) / final_train_loss * 100
            if gap < 10:
                st.success(f"✓ Good generalization (gap: {gap:.1f}%)")
            elif gap < 20:
                st.warning(f"⚠ Moderate overfitting (gap: {gap:.1f}%)")
            else:
                st.error(f"✗ Significant overfitting (gap: {gap:.1f}%)")

        with metric_col2:
            st.subheader("Loss Trends")

            # Calculate moving average
            window = 10
            if len(history['train_loss']) >= window:
                train_ma = pd.Series(history['train_loss']).rolling(window=window).mean()
                val_ma = pd.Series(history['val_loss']).rolling(window=window).mean()

                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(y=train_ma, mode='lines',
                                           name='Train Loss (MA)', line=dict(width=3)))
                fig_ma.add_trace(go.Scatter(y=val_ma, mode='lines',
                                           name='Val Loss (MA)', line=dict(width=3)))
                fig_ma.update_layout(
                    title=f'{window}-Epoch Moving Average',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    template='plotly_white',
                    height=300
                )
                st.plotly_chart(fig_ma, use_container_width=True)

# if __name__ == "__main__":
#     nn_webpage()