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

# --- Streamlit Layout ---

def kmeans_page():
    model, scaler, metadata, history, clustering_data, clustered_df = load_resources()

    if model is not None:

        # --- Cluster Visualization Section ---
        if clustering_data is not None and clustered_df is not None:
            st.markdown("---")
            st.header("🎯 K-Means Cluster Analysis")

            cluster_col1, cluster_col2 = st.columns([1, 2])

            with cluster_col1:
                st.subheader("Clustering Info")
                st.metric("Optimal K (Clusters)", clustering_data['optimal_k'])
                st.metric("Silhouette Score", f"{clustering_data.get('silhouette_score', 'N/A'):.3f}"
                         if 'silhouette_score' in clustering_data else "N/A")

                st.markdown("**Clustering Features:**")
                for feat in clustering_data['features']:
                    st.text(f"• {feat}")

                # Cluster distribution
                cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
                st.markdown("**Cluster Distribution:**")
                for cluster_id, count in cluster_counts.items():
                    st.text(f"Cluster {cluster_id}: {count} counties")

            with cluster_col2:
                st.subheader("Cluster Visualization")

                # 3D scatter plot of clusters
                cluster_features = clustering_data['features']
                if len(cluster_features) >= 3:
                    fig_3d = px.scatter_3d(
                        clustered_df,
                        x=cluster_features[0],
                        y=cluster_features[1],
                        z=cluster_features[2],
                        color='Cluster',
                        title='3D Cluster Visualization',
                        labels={'Cluster': 'Cluster ID'},
                        color_continuous_scale='Viridis'
                    )
                    fig_3d.update_layout(height=500)
                    st.plotly_chart(fig_3d, use_container_width=True)

            # Cluster statistics comparison
            st.subheader("📊 Cluster Characteristics")

            cluster_stats = clustered_df.groupby('Cluster')[clustering_data['features']].mean()

            fig_cluster_bars = go.Figure()
            for feature in clustering_data['features']:
                fig_cluster_bars.add_trace(go.Bar(
                    name=feature,
                    x=[f"Cluster {i}" for i in cluster_stats.index],
                    y=cluster_stats[feature]
                ))

            fig_cluster_bars.update_layout(
                title='Average Feature Values by Cluster',
                xaxis_title='Cluster',
                yaxis_title='Average Value',
                barmode='group',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_cluster_bars, use_container_width=True)

            # Food Insecurity by Cluster
            st.subheader("🍽️ Food Insecurity Rate by Cluster")
            cluster_fi = clustered_df.groupby('Cluster')['Food_Insecurity_Rate'].agg(['mean', 'min', 'max', 'std'])

            fig_fi_cluster = go.Figure()
            fig_fi_cluster.add_trace(go.Bar(
                x=[f"Cluster {i}" for i in cluster_fi.index],
                y=cluster_fi['mean'] * 100,
                error_y=dict(type='data', array=cluster_fi['std'] * 100),
                marker_color='indianred',
                text=(cluster_fi['mean'] * 100).round(2),
                textposition='outside'
            ))

            fig_fi_cluster.update_layout(
                title='Average Food Insecurity Rate by Cluster (with std dev)',
                xaxis_title='Cluster',
                yaxis_title='Food Insecurity Rate (%)',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_fi_cluster, use_container_width=True)