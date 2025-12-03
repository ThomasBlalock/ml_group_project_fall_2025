try:
    import streamlit as st
    from streamlit_image_select import image_select
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit_image_select"])
    import streamlit as st
    from streamlit_image_select import image_select
try:
    import base64
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "base64"])
    import base64
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# """
# To run:
# streamlit run streamlit_app.py
# """

# >>>>>>>>>>>>>>>>>>>> Constants >>>>>>>>>>>>>>>>>>>>>>>>>

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# <<<<<<<<<<<<<<<<<<<< Constants <<<<<<<<<<<<<<<<<<<<<<<<<




# >>>>>>>>>>>>>>>>>>>> Routing >>>>>>>>>>>>>>>>>>>>>>>>>

def app():
    st.set_page_config(layout="wide")
    st.markdown("""
        <div style="background-color:#025464;padding:10px;border-radius:10px">
            <h1 style="color:white;text-align:center;">Taking a Byte Out of Food Insecurity</h1>
            <p style="color:white;text-align:center;font-size:0.9em;font-style:italic;margin-top:-10px;margin-bottom:-10px;">
                Muhammad Amjad &bull; Reed Baumgarner &bull; Thomas Blalock &bull; 
                Helen Corbat &bull; Max Ellingson &bull; Harry Millspaugh &bull; John Twomey
            </p><br>
            <p style="color:white;text-align:center;font-size:0.9em;margin-top:-10px;">
                <b>University of Virginia &bull; Predictive Modeling I &bull; Fall 2025</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs
    h, ds, m = st.tabs(["Home", 
                        "Data Sources", 
                        "Models"], default="Home")
    with ds:
        data_sources()
    with m:
        models()
    with h:
        home()

# <<<<<<<<<<<<<<<<<<<< Routing <<<<<<<<<<<<<<<<<<<<<<<<<



# >>>>>>>>>>>>>>>>>>> Tabs >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def home(): # Home page
    # Interactive ML Models
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Research Questions</h2>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""- What predictors best correlate food insecurity? <- figure this out agfter we make models (put little paragraph why its import and what data we need to can get us there)
- What model can best predict food insecurity?
- What are the most effective interventions to reduce food insecurity?
- Have the causes of food insecurity changed over time?""")
    st.divider()
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Food Insecurity Calculator</h2>
        </div>
        """, unsafe_allow_html=True)
    st.write("We'll put something here where they input data and our models run in the background and give thema prediction.")



def data_sources():
    # Data Sources
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Data Sources</h2>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""**[US Census](https://api.census.gov/data/2022/acs/acs5/variables.html)**
                
insert paragraph explaining it/why it helps us answer our research questions
                
[Map the Meal Gap](https://www.feedingamerica.org/research/map-the-meal-gap/by-county)

inser paragraph explaining it/why it helps us answer our research questions
                """)
    
    # Key Visualizations
    st.divider()
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Key Visualizations</h2>
        </div>
        """, unsafe_allow_html=True)
    st.set_page_config(page_title="Food Insecurity Predictor", layout="wide")
    st.title("🍽️ Food Insecurity Prediction & Analysis Dashboard")

    model, scaler, metadata, history, clustering_data, clustered_df = load_resources()

    if model is not None:

        # --- Data Statistics Section ---
        if clustered_df is not None:
            st.markdown("---")
            st.header("📊 Dataset Statistics & Insights")

            # Summary statistics in columns
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Total Records", len(clustered_df))
            with stat_col2:
                st.metric("Avg Food Insecurity", f"{clustered_df['Food_Insecurity_Rate'].mean():.2%}")
            with stat_col3:
                st.metric("Avg Poverty Rate", f"{clustered_df['POVERTY_RATE'].mean():.1f}%")
            with stat_col4:
                st.metric("Avg Unemployment", f"{clustered_df['UNEMPLOYMENT_RATE'].mean():.1f}%")

            # Feature Distributions
            st.subheader("📈 Feature Distributions")

            feature_options = ['POVERTY_RATE', 'UNEMPLOYMENT_RATE', 'SNAP_RECEIPT_RATE',
                             'MEDIAN_HOUSEHOLD_INCOME', 'Food_Insecurity_Rate']
            selected_feature = st.selectbox("Select feature to visualize:", feature_options)

            fig_dist = px.histogram(clustered_df, x=selected_feature, nbins=30,
                                   title=f'Distribution of {selected_feature}',
                                   color='Cluster' if 'Cluster' in clustered_df.columns else None,
                                   marginal='box')
            fig_dist.update_layout(template='plotly_white', showlegend=True)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Correlation Heatmap
            st.subheader("🔥 Feature Correlation Matrix")
            numeric_features = ['MEDIAN_HOUSEHOLD_INCOME', 'POVERTY_RATE', 'UNEMPLOYMENT_RATE',
                               'SNAP_RECEIPT_RATE', 'POP_16_PLUS', 'HOUSEHOLDS_TOTAL',
                               'Food_Insecurity_Rate']

            corr_data = clustered_df[numeric_features].corr()

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu_r',
                zmid=0,
                text=corr_data.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            fig_corr.update_layout(
                title='Feature Correlation Heatmap',
                template='plotly_white',
                height=600
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("Red indicates positive correlation, blue indicates negative correlation.")


def get_img_as_base64(file_path): # Helper fn
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def models():
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Machine Learning Models</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Buttons For Models
    model_names = ["Linear Model", "KNN", "K-Means", "PCA", "Neural Network"]
    model_images = [
        os.path.join(ROOT_DIR, "assets", "linear_model.jpg"), #0
        os.path.join(ROOT_DIR, "assets", "knn_model.jpg"), #1
        os.path.join(ROOT_DIR, "assets", "kmeans_model.png"), #2
        os.path.join(ROOT_DIR, "assets", "pca_model.png"), #3
        os.path.join(ROOT_DIR, "assets", "neural_network_model.jpg") #4
    ]

    selected_model = image_select(
        label="",
        images=model_images,
        captions=model_names,
        use_container_width=False,
        index=0,
        return_value="original"
    )
    if selected_model == model_images[0]:
        st.subheader("Linear Model")
        linear_page()

    elif selected_model == model_images[1]:
        st.subheader("K-Nearest Neighbors")
        knn_page()

    elif selected_model == model_images[2]:
        st.subheader("K-Means Clustering")
        kmeans_page()

    elif selected_model == model_images[3]:
        st.subheader("Principal Component Analysis")
        pca_page()

    elif selected_model == model_images[4]:
        st.subheader("Neural Network")
        nn_page()


# <<<<<<<<<<<<<<<<<<<<<<<< Tabs <<<<<<<<<<<<<<<<<<<<<<<<<<<


# >>>>>>>>>>>>>>>>>>> Neural Network Page >>>>>>>>>>>>>>>>>>> 

# --- NN Constants ---

NN_ARTIFACTS_DIR = os.path.join(ROOT_DIR, "models", "neural_net", "model_artifacts")

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

def make_prediction(model, scaler, metadata, input_data):
    df_input = pd.DataFrame([input_data])
    for col in metadata['categorical_cols']:
        df_input[col] = df_input[col].astype(str)
        
    df_encoded = pd.get_dummies(df_input, columns=metadata['categorical_cols'], drop_first=True)
    df_encoded = df_encoded.reindex(columns=metadata['feature_columns'], fill_value=0)
    X_numeric = df_encoded[metadata['numeric_cols']].values
    X_categorical = df_encoded[metadata['encoded_cat_cols']].values.astype(float)
    X_numeric_scaled = scaler.transform(X_numeric)
    X_final = np.hstack([X_numeric_scaled, X_categorical])
    tensor_input = torch.FloatTensor(X_final)
    with torch.no_grad():
        prediction = model(tensor_input).item()
        
    return prediction

# --- Streamlit Layout ---

def nn_page():

    # Get Model Resources
    mmetadata = joblib.load(os.path.join(NN_ARTIFACTS_DIR, "model_metadata.save"))
    scaler = joblib.load(os.path.join(NN_ARTIFACTS_DIR, "scaler.save"))
    input_dim = len(metadata['feature_columns'])
    model = FoodSecurityFFNN(input_dim)
    model.load_state_dict(torch.load(os.path.join(NN_ARTIFACTS_DIR, "food_security_model.pth"), map_location=torch.device('cpu')))
    model.eval()
    with open(os.path.join(NN_ARTIFACTS_DIR, "training_history.json"), 'r') as f:
        history = json.load(f)

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

# <<<<<<<<<<<<<<<<<<< Neural Network Page <<<<<<<<<<<<<<<<<<<<<



# >>>>>>>>>>>>>>>>>>> K-Means Page >>>>>>>>>>>>>>>>>>>>>>>>>

def kmeans_page():
    clustering_data = joblib.load(os.path.join(NN_ARTIFACTS_DIR, "clustering_model.save"))
    clustered_df = pd.read_csv(os.path.join(NN_ARTIFACTS_DIR, "data_with_clusters.csv"))

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

# <<<<<<<<<<<<<<<<<<<<<<< K-Means Page <<<<<<<<<<<<<<<<<<<<<<<



app()