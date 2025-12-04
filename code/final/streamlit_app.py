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

ROOT_DIR = os.getcwd()

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

    clustered_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "data_w_clusters.csv"))
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

# >>>>>>>>>>>>>>>>>>>>>>>> Linear Page >>>>>>>>>>>>>>>>>>>>>

def linear_page():
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Linear Model</h2>
        </div>
        """, unsafe_allow_html=True)

# <<<<<<<<<<<<<<<<<<<<<<<< Linear Page <<<<<<<<<<<<<<<<<<<<<<<


# >>>>>>>>>>>>>>>>>>>>>>>> KNN Page >>>>>>>>>>>>>>>>>>>>>>>>>

def knn_page():
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">K-Nearest Neighbors</h2>
        </div>
        """, unsafe_allow_html=True)

# <<<<<<<<<<<<<<<<<<<<<<<< KNN Page <<<<<<<<<<<<<<<<<<<<<<<<<


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
    metadata = joblib.load(os.path.join(NN_ARTIFACTS_DIR, "model_metadata.save"))
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
        cat_col1, cat_col3, cat_col4 = st.columns(3)

        # Categorical Dropdowns (Populated from Metadata)
        with cat_col1:
            input_data['State'] = st.selectbox("State", metadata['unique_values']['State'])
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

        # Plotly Interactive Graph
        history['log_train_loss'] = np.log(history['train_loss'])
        history['log_val_loss'] = np.log(history['val_loss'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history['log_train_loss'], mode='lines', name='Training Loss'))
        fig.add_trace(go.Scatter(y=history['log_val_loss'], mode='lines', name='Validation Loss'))
        
        fig.update_layout(
            xaxis_title='Epoch',
            yaxis_title='Log Mean Squared Error',
            template='plotly_white',
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("Lower loss indicates better model performance. The gap between Training and Validation shows how well the model generalizes.")


        METRICS_FILE = "models/neural_net/model_artifacts/model_metrics.json"
        with open(METRICS_FILE, 'r') as f:
            data = json.load(f)

        st.title("🍔 Food Insecurity Model Evaluation")
    
        if not data:
            return

        # --- Header Metrics ---
        st.markdown("### Top Level Performance")
        col1, col2, col3 = st.columns(3)
        
        # R2 Score
        col1.metric("Regression R²", f"{data['r2_score']:.4f}")
        
        # Accuracy (extracted from classification report)
        accuracy = data['classification_report']['accuracy']
        col2.metric("Classification Accuracy", f"{accuracy:.2%}")
        
        # Macro F1
        macro_f1 = data['classification_report']['macro avg']['f1-score']
        col3.metric("Macro F1-Score", f"{macro_f1:.4f}")

        st.divider()

        # --- Visualization Layout ---
        col_viz, col_data = st.columns([1, 1])

        with col_viz:
            st.subheader("Confusion Matrix")
            
            # Reconstruct DataFrame for Seaborn
            labels = data['labels']
            cm_matrix = data['confusion_matrix']
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                xticklabels=labels, 
                yticklabels=labels,
                cbar=False,
                ax=ax
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Prediction vs Ground Truth")

            # Labels Color
            text_color = 'white'
            plt.xlabel("Predicted", color=text_color)
            plt.ylabel("Actual", color=text_color)
            plt.title("Prediction vs Ground Truth", color=text_color)
            ax.tick_params(axis='x', colors=text_color)
            ax.tick_params(axis='y', colors=text_color)
            
            # Make the background transparent for streamlit
            fig.patch.set_alpha(0)
            st.pyplot(fig)

        with col_data:
            st.subheader("Detailed Classification Report")
            
            # Convert dictionary to DataFrame
            report_df = pd.DataFrame(data['classification_report'])
            del report_df['weighted avg']
            report_df = report_df.rename(columns={'macro avg': 'avg'})
            report_df = report_df.transpose()
            
            # Remove the 'accuracy' row for the main table (it messes up the column alignment usually)
            # We displayed accuracy at the top already
            report_clean = report_df.drop('accuracy', errors='ignore')

            # Formatting Logic
            def highlight_low_scores(val):
                color = '#ff4b4b' if val < 0.5 else ''
                return f'color: {color}'

            # Display as a styled dataframe
            st.dataframe(
                report_clean.style
                .format("{:.3f}")
                .map(highlight_low_scores, subset=['precision', 'recall', 'f1-score'])
                .background_gradient(subset=['f1-score'], cmap="Greens", vmin=0, vmax=1),
                use_container_width=True,
                height=400
            )
            
            st.info("💡 **Note:** 'Support' refers to the number of actual occurrences of the class in the dataset.")

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


# >>>>>>>>>>>>>>>>>>>>>> PCA Page >>>>>>>>>>>>>>>>>>>>>>>>>>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

def pca_page():
    data = pd.read_csv(os.path.join(ROOT_DIR, "data", "data.csv"))
    numericCols = [
        'Food_Insecurity_Rate', 'Num_Food_Insecure_Persons', 'Low_Threshold_State',
        'High_Threshold_State', 'Pct_FI_Below_Low_Threshold', 'Pct_FI_Between_Thresholds',
        'Pct_FI_Above_High_Threshold', 'Child_Food_Insecurity_Rate',
        'Num_Food_Insecure_Children', 'Pct_FI_Children_Below_185FPL',
        'Pct_FI_Children_Above_185FPL', 'MEDIAN_HOUSEHOLD_INCOME', 'POP_POVERTY_DETERMINED',
        'POP_BELOW_POVERTY', 'POP_16_PLUS', 'POP_UNEMPLOYED', 'HOUSEHOLDS_TOTAL',
        'HOUSEHOLDS_SNAP', 'POVERTY_RATE', 'UNEMPLOYMENT_RATE', 'SNAP_RECEIPT_RATE'
    ]
    subData = data[numericCols].replace('-*', np.nan).dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(subData)
    pca = PCA()
    xPCA = pca.fit_transform(scaled)

    # Scree plot to choose number of components
    plt.figure(figsize=(8,5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='x', linestyle='--')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA - Explained Variance Ratio")
    st.write("Scree Plot")
    st.pyplot(plt)
    st.write("Based on the scree plot, we will use 4 components.")

    pcaDF = pd.load_csv(os.path.join(ROOT_DIR, "data", "pcaDF.csv"))

    # Visualize clusters in PCA space
    sns.pairplot(pcaDF, palette='Set2')
    plt.suptitle("KMeans Clusters on PCA Components", y=1.02)
    st.write("Pairplot of KMeans Clusters on PCA Components")
    st.pyplot(plt)

    # Pipeline with scaler + KMeans
    feature_cols = numericCols
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=2, init='k-means++', n_init=15, random_state=67))
    ])
    pipe.fit(subData[feature_cols])

    # Extract labels and centroids
    subData['cluster'] = pipe['kmeans'].labels_
    centroids_scaled = pipe['kmeans'].cluster_centers_
    centroids_original = pipe['scaler'].inverse_transform(centroids_scaled)

    # Centroids DataFrame
    pd.set_option('display.max_columns', None)
    cent_df = pd.DataFrame(centroids_original, columns=feature_cols)

    st.write("Centroids DataFrame")
    st.write(cent_df.T)


    st.markdown("""
    ## Cluster Analysis

The two clusters represent distinct socioeconomic and food-security profiles across the units in the dataset. Cluster 0 is characterized by substantially lower absolute counts of food-insecure individuals and households, lower total population and household totals, and generally lower poverty-related burdens. For example, the centroid for Num_Food_Insecure_Persons (~9,600), POP_BELOW_POVERTY (~9,800), and HOUSEHOLDS_TOTAL (~25,700) suggests smaller, less densely populated communities with relatively moderated levels of economic hardship. The food insecurity rate (~0.145) and child food insecurity rate (~0.219) are meaningful but not extreme, and median household income (~$46,400) is modest yet notably higher than typical high-poverty geographies. Overall, Cluster 0 represents moderately food-insecure, lower-population regions with more stable economic indicators.

In contrast, Cluster 1 reflects a dramatically different profile, with an order-of-magnitude increase in population and economic strain. These areas exhibit extremely high counts of food-insecure persons (~425,000), food-insecure children (~150,000), and households receiving SNAP (~121,000). Poverty and unemployment burdens are also substantially higher in absolute size, and median household income (~$54,200) is slightly higher than Cluster 0 but does not compensate for the much larger populations living below poverty. Interestingly, the rates of food insecurity and child food insecurity are similar to those of Cluster 0, but the total scale of affected individuals is vastly larger. Thus, Cluster 1 captures high-population, high-need metropolitan or regional centers where structural poverty affects a far larger volume of residents, even when rate-based indicators appear comparable.

Taken together, these clusters differentiate not merely by intensity of food insecurity in percentage terms, but by structural magnitude—Cluster 0 represents smaller, moderately burdened communities, whereas Cluster 1 captures large-scale, high-need population centers where social assistance demand, economic vulnerability, and food insecurity exist at a substantially greater scale. This distinction is especially relevant for resource allocation: policies optimized for Cluster 1 must address volume and infrastructure capacity, whereas interventions for Cluster 0 may focus on rural access, localized service gaps, and targeted support.
    """)

# <<<<<<<<<<<<<<<<<<<<<<< PCA Page <<<<<<<<<<<<<<<<<<<<<<<



# >>>>>>>>>>>>>>>>>>>>>> KMeans Page >>>>>>>>>>>>>>>>>>>>>>>>>>

def kmeans_page():
    data = pd.read_csv(os.path.join(ROOT_DIR, "data", "data.csv"))
    numericCols = [
        'Food_Insecurity_Rate', 'Num_Food_Insecure_Persons', 'Low_Threshold_State',
        'High_Threshold_State', 'Pct_FI_Below_Low_Threshold', 'Pct_FI_Between_Thresholds',
        'Pct_FI_Above_High_Threshold', 'Child_Food_Insecurity_Rate',
        'Num_Food_Insecure_Children', 'Pct_FI_Children_Below_185FPL',
        'Pct_FI_Children_Above_185FPL', 'MEDIAN_HOUSEHOLD_INCOME', 'POP_POVERTY_DETERMINED',
        'POP_BELOW_POVERTY', 'POP_16_PLUS', 'POP_UNEMPLOYED', 'HOUSEHOLDS_TOTAL',
        'HOUSEHOLDS_SNAP', 'POVERTY_RATE', 'UNEMPLOYMENT_RATE', 'SNAP_RECEIPT_RATE'
    ]
    subData = data[numericCols].replace('-*', np.nan).dropna()
    # Pipeline with scaler + KMeans
    feature_cols = numericCols
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=2, init='k-means++', n_init=15, random_state=67))
    ])
    pipe.fit(subData[feature_cols])

    # Extract labels and centroids
    subData['cluster'] = pipe['kmeans'].labels_

    plt.figure(figsize=(8,6))
    plt.scatter(subData['Child_Food_Insecurity_Rate'], 
                subData['Food_Insecurity_Rate'], 
                c=subData['cluster'].astype(int), cmap='rainbow', alpha=0.2)
    plt.xlabel("Child Food Insecurity Rate")
    plt.ylabel("Food Insecurity Rate")
    plt.title("KMeans Clusters (Original Features)")
    st.pyplot(plt)

    K_values = list(range(1,15))
    wcss = []
    for k in K_values:
        pipe.set_params(kmeans__n_clusters=k)
        pipe.fit(subData[feature_cols])
        wcss.append(pipe['kmeans'].inertia_)

    fig = px.line(x=K_values, y=wcss, markers=True,
                title="Elbow Plot",
                labels={"x":"Number of Clusters", "y":"WCSS"})
    st.pyplot(fig)

    sil_scores = []
    K_values_sil = list(range(2,15))
    for k in K_values_sil:
        pipe.set_params(kmeans__n_clusters=k)
        pipe.fit(subData[feature_cols])
        labels = pipe['kmeans'].labels_
        sil_scores.append(silhouette_score(subData[feature_cols], labels))

    fig = px.line(x=K_values_sil, y=sil_scores, markers=True,
                title="Silhouette Scores",
                labels={"x":"Number of Clusters", "y":"Silhouette Score"})
    st.pyplot(fig)

    def predict_kmeans_cluster(new_row, pipeline, feature_cols):
        """
        Predict KMeans cluster for a new row using an existing pipeline.

        Parameters:
        new_row : dict
            Dictionary of feature values (keys must match feature_cols)
        pipeline : sklearn.pipeline.Pipeline
            Fitted pipeline
        feature_cols : list
            List of column names used in fitting

        Returns:
        cluster_label : int
        """
        df = pd.DataFrame([new_row], columns=feature_cols)
        return pipeline.predict(df)[0]

# <<<<<<<<<<<<<<<<<<<<<<< KMeans Page <<<<<<<<<<<<<<<<<<<<<<<




app()