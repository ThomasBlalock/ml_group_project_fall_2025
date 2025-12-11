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
import PIL
from PIL import Image

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
            <h1 style="color:white;text-align:center;">Predictive Modelling Approaches to U.S. Food Insecurity</h1>
            <p style="color:white;text-align:center;font-size:0.9em;font-style:italic;margin-top:-10px;margin-bottom:-10px;">
                Muhammad Amjad &bull; Reed Baumgardner &bull; Thomas Blalock &bull; 
                Helen Corbat &bull; Max Ellingsen &bull; Harry Millspaugh &bull; John Twomey
            </p><br>
            <p style="color:white;text-align:center;font-size:0.9em;margin-top:-10px;">
                <b>University of Virginia &bull; Predictive Modeling I &bull; Fall 2025</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs
    h, ds, m, c = st.tabs(["Home", 
                        "Data Sources", 
                        "Models",
                        "Conclusion"], default="Home")
    with ds:
        data_sources()
    with m:
        models()
    with h:
        home()
    with c:
        conclusion()

# <<<<<<<<<<<<<<<<<<<< Routing <<<<<<<<<<<<<<<<<<<<<<<<<



# >>>>>>>>>>>>>>>>>>> Tabs >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def conclusion():
    future_work = """
        # Conclusions & Future Directions

        ## Conclusions
        Our comparative analysis highlights that food insecurity is a highly non-linear phenomenon deeply entrenched in local socioeconomic conditions. While the **Neural Network** achieved the highest $R^2$ and a ~70% F-Score, indicating a robust ability to model complex interactions, our interpretable models (GLM and K-Means) provided critical policy insights:

        * **Spatial Autocorrelation:** Counties that are geographically adjacent tend to feature similar food insecurity profiles. This suggests that hunger is rarely an isolated local failure but often a regional systemic issue, necessitating **greater regional or statewide policy cooperation** rather than siloed county-level efforts.
        * **Bifurcated Policy Strategy:** Our clustering analysis suggests a need to split implementation strategies based on community type:
            * **Urban Areas (Cluster B):** Policy should prioritize **volume and infrastructure capacity** to handle high absolute numbers of food-insecure households.
            * **Rural Areas (Cluster A):** Policy should prioritize **targeted support** to address localized service gaps and logistical isolation.
        * **The SNAP Threshold:** Analysis identifies the "SNAP threshold" (the income level required to qualify for aid) as a critical lever. We recommend **lowering the SNAP threshold requirements for high-risk areas** identified in our analysis to increase coverage for working-poor households that currently fall into the "gap" (ineligible for aid but unable to afford food).

        ## Future Research
        To further refine these models for real-world application, future iterations of this project will focus on three key areas:

        1.  **Expanded Feature Space:** We aim to incorporate non-demographic predictors to capture the "micro" environment of food access. This includes **family structural components** (single-parent household dynamics), **community infrastructure attributes** (transit access, grocery store density), and **state agricultural research output**.
        2.  **Temporal Validation:** We will validate and iterate our models using **post-2018 data** as it becomes available. Testing against newer Census measures will help determine if our models remain robust in shifting economic climates.
        3.  **Explainable AI (XAI):** While the Neural Network performed best, it remains a "black box." We plan to perform an **XAI analysis** (e.g., using SHAP values) to transparently map how the network weighs specific inputs. This is crucial for validating the model's behavior and ensuring it is ethical and suitable for use in high-stakes policy decisions.
        """
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Conclusion & Future Work</h2>
            {}
        </div>
        """.format(future_work), unsafe_allow_html=True)
    


def home(): # Home page
    # Interactive ML Models
    st.markdown("""
        # Project Overview
        **Forecasting Resilience: A Comparative Analysis of Predictive Models for County-Level Food Insecurity**

        ## Introduction
        Food insecurity remains a pervasive and complex challenge in the United States, affecting millions of households despite national economic growth. While the USDA plays a critical role in subsidizing food access and nutrition education, the drivers of hunger vary significantly across local geographies. Effective intervention requires more than broad federal aid; it demands precise, data-driven insights into where need is greatest and *why*.

        This project aims to bridge the gap between national statistics and local realities. We utilized a comprehensive dataset spanning **2011–2018**, combining proprietary food insecurity data acquired via direct request from **Feeding America** with socioeconomic indicators from the **U.S. Census Bureau**. By supplementing "ground truth" hunger metrics with granular income, employment, and poverty features, we rigorously tested multiple machine learning algorithms—from General Linear Models to Neural Networks—to model the landscape of American food insecurity.

        ## Research Questions
        Our analysis is guided by four core inquiries designed to inform both statistical modeling and public policy:

        * **Model Efficacy:** Which predictive architecture (GLM, KNN, or Neural Network) yields the lowest error rates and robust performance when forecasting county-level food insecurity?
        * **Feature Importance:** Using interpretable models like GLM, which socioeconomic variables (e.g., poverty rate, SNAP thresholds) emerge as the strongest statistical predictors of hunger?
        * **Clustering Methods:** Can U.S. counties be grouped into distinct "typologies" (e.g., via K-Means) to allow for tailored policy interventions rather than a "one-size-fits-all" approach?
        * **Signal Detection:** Is there a strong spatial signal to food insecurity? Can we accurately predict a county's risk category simply by analyzing its most similar neighbors (KNN)?
        """)
        #     st.markdown("""- What predictors best correlate food insecurity? <- figure this out agfter we make models (put little paragraph why its import and what data we need to can get us there)
        # - What model can best predict food insecurity?
        # - What are the most effective interventions to reduce food insecurity?
        # - Have the causes of food insecurity changed over time?""")
    st.divider()
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Food Insecurity Calculator</h2>
        </div>
        """, unsafe_allow_html=True)
    calculator_section()
    calculator_knn_section()



def data_sources():
    # Data Sources
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Data Sources</h2>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""**[US Census](https://api.census.gov/data/2022/acs/acs5/variables.html)**
                
For this project, we utilize the American Community Survey (ACS) to build the foundation of our predictive models. This dataset serves as our primary source for independent variables (predictors), giving us the granular, county-level socioeconomic data necessary to understand the root causes of food insecurity.

Specifically, we use Census data to capture the economic stability of a county through variables like MEDIAN_HOUSEHOLD_INCOME, POVERTY_RATE, and UNEMPLOYMENT_RATE. Beyond these standard economic indicators, the Census allows us to dig deeper into structural vulnerability by analyzing HOUSEHOLDS_SNAP (SNAP participation) and POP_POVERTY_DETERMINED (the raw population for whom poverty status is known). By feeding these specific demographic and economic inputs into our models, we can test which community characteristics—whether it be a lack of income, high unemployment, or reliance on government assistance—are the strongest statistical signals for predicting hunger.
                
[Map the Meal Gap](https://www.feedingamerica.org/research/map-the-meal-gap/by-county)

While the Census provides the predictors, the Map the Meal Gap (MMG) dataset provides our target variables (ground truth). Since the USDA does not measure food insecurity directly at the county level, we rely on MMG’s validated estimates to train and test our models.

This dataset allows us to answer our core research questions by providing the actual Food_Insecurity_Rate and Num_Food_Insecure_Persons that we are trying to predict. Crucially, MMG also helps us distinguish between simple poverty and actual food hardship by providing unique cost-of-living metrics like Cost_Per_Meal and the Annual_Food_Budget_Shortfall.

Furthermore, MMG enables us to perform more nuanced analysis beyond just a single "hunger rate." With variables like Pct_FI_Below_Low_Threshold and Child_Food_Insecurity_Rate, we can separate our analysis to see if different models are needed to predict insecurity for children versus the general population, or for families who qualify for federal aid versus those who don't.
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

        artifacts_dir = os.path.join(ROOT_DIR, "models", "linear", "artifacts")
        def render_artifact(filename, header, description=""):
            filepath = os.path.join(artifacts_dir, filename)
            if os.path.exists(filepath):
                st.markdown(f"### {header}")
                if description:
                    st.caption(description)
                image = Image.open(filepath)
                st.image(image, use_container_width=True) # use_container_width replaces use_column_width in newer Streamlit
                st.divider()
            else:
                # Optional: handle missing files quietly or with a warning
                pass

        # GEOGRAPHIC CONTEXT
        st.info("Geographic Analysis")

        left_data, right_data = st.columns(2)
        
        with left_data:
            render_artifact("child_food_i_map.png", "Child Food Insecurity Map", 
                            "Geographic distribution of the target variable.")
        with right_data:
            render_artifact("cost_per_meal_map.png", "Cost Per Meal Map", 
                            "Geographic distribution of meal costs.")

        # # Make button to generate report
        # if st.button("Generate Data Profiling Report"):
        #     with open(os.path.join(ROOT_DIR, "data_engineering", "data_profiling.html"), "r") as f:
        #         profile_report = f.read()

        #     st.markdown(profile_report, unsafe_allow_html=True)


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
    # Main Page Header
    st.markdown("""
        <div style="padding:10px;border-radius:10px;margin-bottom:20px">
            <h1 style="color:white;text-align:center;">Linear Model Analysis</h1>
        </div>
        """, unsafe_allow_html=True)

    # Base path for artifacts
    # Ensure ROOT_DIR is defined in your global scope or imports
    artifacts_dir = os.path.join(ROOT_DIR, "models", "linear", "artifacts")

    # Helper function to display images safely with headers
    def render_artifact(filename, header, description=""):
        filepath = os.path.join(artifacts_dir, filename)
        if os.path.exists(filepath):
            st.markdown(f"### {header}")
            if description:
                st.caption(description)
            image = Image.open(filepath)
            st.image(image, use_container_width=True) # use_container_width replaces use_column_width in newer Streamlit
            st.divider()
        else:
            # Optional: handle missing files quietly or with a warning
            pass

    # --- SECTION 3: CORRELATIONS & RELATIONSHIPS ---
    st.info("Feature Correlations")
    left, right = st.columns(2)
    
    with left:
        render_artifact("corr_plot.png", "General Correlation Matrix")
    with right:
        render_artifact("corr_plot_num.png", "Numeric Correlation Detail")

    # --- SECTION 4: MODEL RESULTS ---
    st.success("OLS Model Results")
    st.write("Comparison of the initial model vs. the refined model.")
    left1, right1 = st.columns(2)
    with left1:
        render_artifact("OLS_1_results.png", "Model 1: OLS Summary")
    with right1:
        render_artifact("OLS_2_results.png", "Model 2: OLS Summary")

    render_artifact("child_food_i_by_income.png", "Insecurity vs. Income", 
                    "Visualizing the relationship between income levels and food insecurity.")

# <<<<<<<<<<<<<<<<<<<<<<<< Linear Page <<<<<<<<<<<<<<<<<<<<<<<


# >>>>>>>>>>>>>>>>>>>>>>>> KNN Page >>>>>>>>>>>>>>>>>>>>>>>>>
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA


def knn_page():
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">K-Nearest Neighbors</h2>
        </div>
        """, unsafe_allow_html=True)

    results_df = pd.read_csv("models/knn/artifacts/knn_cv_results.csv")
    best_k = 7
    best_score = 0.6317147
    fig = px.line(
        results_df,
        x="k",
        y="mean_score",
        title=f"Cross-Validated Balanced Accuracy vs. K (best k = 7)",
        markers=True,
        labels={"k": "Number of Neighbors (k)", "mean_score": "Mean CV Balanced Accuracy"}
    )


    fig.add_scatter(
        x=[best_k],
        y=[best_score],
        mode="markers+text",
        text=[f"Best k = {best_k}"],
        textposition="top center",
        name="Best k"
    )

    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig)

    # data = pd.read_csv('./data/data.csv')
    # data = data.dropna(subset=[col for col in data.columns if col != "Pct_FI_Between_Thresholds"])
    # bins = [0, 0.115, 0.138, 0.164, 1]  # 1 is just a safe upper bound
    # labels = ["Low", "Moderate", "Elevated", "High"]

    # data["FI_Category"] = pd.cut(
    #     data["Food_Insecurity_Rate"],
    #     bins=bins,
    #     labels=labels,
    #     include_lowest=True
    # )

    # data["FI_Category"].value_counts()
    # y=data['FI_Category']
    # X = data[['MEDIAN_HOUSEHOLD_INCOME',
    #         'POP_POVERTY_DETERMINED',
    #         'POP_BELOW_POVERTY',
    #         'POP_16_PLUS',
    #         'POP_UNEMPLOYED',
    #         'HOUSEHOLDS_TOTAL',
    #         'HOUSEHOLDS_SNAP',
    #         'POVERTY_RATE',
    #         'UNEMPLOYMENT_RATE',
    #         'SNAP_RECEIPT_RATE',
    #         'Cost_Per_Meal',
    #         'Annual_Food_Budget_Shortfall']]
    # county_series = data["County"]
    # state_series = data["State"]
    # fips_series = data["FIPS"]

    # X_train, X_test, y_train, y_test, county_train, county_test, state_train, state_test, fips_train, fips_test = train_test_split(
    #     X, y, county_series, state_series, fips_series,
    #     test_size=0.25,
    #     random_state=42,
    #     stratify=y
    # )

    # pipe = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("knn", KNeighborsClassifier(weights="distance"))
    # ])

    # param_grid = {"knn__n_neighbors": range(1, 41, 2)}

    # grid = GridSearchCV(pipe, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1)
    # grid.fit(X_train, y_train)

    # results_df = pd.DataFrame(grid.cv_results_)

    # results_df["k"] = results_df["param_knn__n_neighbors"]
    # results_df["mean_score"] = results_df["mean_test_score"]
    # results_df.to_csv("models/knn/artifacts/knn_cv_results.csv", index=False)

    # pipe2 = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("knn", KNeighborsClassifier(n_neighbors=best_k,
    #     weights="distance"))
    # ])

    # pipe2.fit(X_train, y_train)
    # y_pred = pipe2.predict(X_test)

    # acc = accuracy_score(y_test, y_pred)
    # bal_acc = balanced_accuracy_score(y_test, y_pred)

    st.info("Accuracy: 0.640 | Balanced accuracy: 0.630")

    # from sklearn.metrics import confusion_matrix

    # cm = confusion_matrix(y_test, y_pred, labels=pipe2.classes_)

    # cm_df = pd.DataFrame(
    #     cm,
    #     index=[f"Actual {c}" for c in pipe2.classes_],
    #     columns=[f"Predicted {c}" for c in pipe2.classes_]
    # )
    # cm_df.to_csv(os.path.join(ROOT_DIR, "models", "knn", "artifacts", "knn_confusion_matrix.csv"))
    cm_df = pd.read_csv(os.path.join(ROOT_DIR, "models", "knn", "artifacts", "knn_confusion_matrix.csv"))

    st.write(cm_df)

    # scaler_pca = StandardScaler()
    # X_scaled_full = scaler_pca.fit_transform(X)

    # pca = PCA(n_components=2, random_state=42)
    # X_pca = pca.fit_transform(X_scaled_full)

    # pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    # pca_df["FI_Category"] = y.values
    # pca_df.to_csv(os.path.join(ROOT_DIR, "models", "knn", "artifacts", "knn_pca_df.csv"), index=False)
    pca_df = pd.read_csv(os.path.join(ROOT_DIR, "models", "knn", "artifacts", "knn_pca_df.csv"))

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="FI_Category",
        title="PCA (2D) of MMG Features Labeled by FI Category",
        hover_data=["FI_Category"],
    )
    st.plotly_chart(fig)



def predict_county(county_name, state_name=None):
    # 1. Load and Clean Data
    data = pd.read_csv('./data/data.csv')
    data = data.dropna(subset=[col for col in data.columns if col != "Pct_FI_Between_Thresholds"])
    
    # 2. Setup Bins and Categories
    bins = [0, 0.115, 0.138, 0.164, 1]  # 1 is just a safe upper bound
    labels = ["Low", "Moderate", "Elevated", "High"]
    
    data["FI_Category"] = pd.cut(
        data["Food_Insecurity_Rate"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    
    # 3. Define Features (X) and Target (y)
    y = data['FI_Category']
    X = data[['MEDIAN_HOUSEHOLD_INCOME',
            'POP_POVERTY_DETERMINED',
            'POP_BELOW_POVERTY',
            'POP_16_PLUS',
            'POP_UNEMPLOYED',
            'HOUSEHOLDS_TOTAL',
            'HOUSEHOLDS_SNAP',
            'POVERTY_RATE',
            'UNEMPLOYMENT_RATE',
            'SNAP_RECEIPT_RATE',
            'Cost_Per_Meal',
            'Annual_Food_Budget_Shortfall']]

    category_ranges = {
        "Low": "[0.000, 0.115]",
        "Moderate": "(0.115, 0.138]",
        "Elevated": "(0.138, 0.164]",
        "High": "(0.164, 1.000]"
    }

    # Mean numeric FI rate for each category (used as predicted numeric)
    cat_mean_rate = (
        data.groupby("FI_Category")["Food_Insecurity_Rate"]
        .mean()
        .to_dict()
    )

    # 4. Train Model (on entire dataset as per your snippet)
    pipe2 = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance"))
    ])
    pipe2.fit(X, y)

    # 5. Filter Data for specific County/State
    df = data.copy()
    mask = df["County"].str.contains(county_name, case=False, na=False)
    if state_name:
        mask &= df["State"].str.contains(state_name, case=False, na=False)

    matches = df[mask]

    if matches.empty:
        print("No matching county found.")
        return []

    # 6. Iterate through all matching years (rows)
    results_list = []
    
    print(f"Found {len(matches)} records for {county_name}...\n")

    for index, row in matches.iterrows():
        # Extract features for this specific year/row
        X_row = row[X.columns].to_frame().T
        actual = row["Food_Insecurity_Rate"]
        county = row["County"]
        state = row["State"]
        # Try to get Year if it exists, otherwise use index
        year = row.get("YEAR", "Unknown Year") 

        # --- MODEL PREDICTION ---
        pred_cat = pipe2.predict(X_row)[0]
        pred_cat_str = str(pred_cat)
        pred_rate = cat_mean_rate[pred_cat]

        # Actual category
        actual_cat = pd.cut(
            pd.Series([actual]),
            bins=bins,
            labels=labels,
            include_lowest=True
        ).iloc[0]
        actual_cat_str = str(actual_cat)

        # Print output for console debugging
        print(f"--- {year} ---")
        print(f"Pred: {pred_cat_str} ({pred_rate:.3f}) | Actual: {actual:.3f}")

        # Append to results list
        results_list.append({
            "County": county,
            "State": state,
            "Year": year,  # Added Year to dictionary
            "Predicted FI Category": pred_cat_str,
            "Predicted FI Rate": pred_rate,
            "Category Range": category_ranges[pred_cat_str],
            "Actual FI Rate": actual,
            "Actual FI Category": actual_cat_str
        })

    return results_list


def display_prediction_card(data_list):
    """
    Accepts a list of dictionaries (one per year) and displays them.
    """
    if not data_list:
        st.error("No data available to display.")
        return

    # Loop through the list of results
    for data in data_list:
        
        # Container to hold the styling
        with st.container(border=True):
            
            # Header: County, State AND Year
            year_display = f"({data.get('Year', 'N/A')})"
            st.markdown(f"### 📍 {data['County']}, {data['State']} {year_display}")
            st.divider()

            # Layout: 3 Columns for Metrics
            col1, col2, col3 = st.columns(3)

            # Calculate Error (Delta)
            pred_val = data['Predicted FI Rate']
            actual_val = data['Actual FI Rate']
            
            # Formatting assumption: If rate is 0.15, display as 15.0%
            # If your data is already 15.0, remove the (* 100) below.
            display_pred = pred_val * 100
            display_actual = actual_val * 100
            
            delta = round(display_pred - display_actual, 2)

            # Column 1: The Prediction
            with col1:
                st.metric(
                    label="Predicted Rate",
                    value=f"{display_pred:.1f}%",
                    delta=f"{delta}% Error",
                    delta_color="inverse" 
                )
                st.caption(f"Range: {data['Category Range']}")

            # Column 2: The Ground Truth
            with col2:
                st.metric(
                    label="Actual Rate",
                    value=f"{display_actual:.1f}%"
                )

            # Column 3: Category Match Status
            with col3:
                is_match = data['Predicted FI Category'] == data['Actual FI Category']
                
                st.markdown("**Category Accuracy**")
                if is_match:
                    st.success(f"✅ Match: {data['Predicted FI Category']}")
                else:
                    st.error(f"❌ Missed")
                    st.caption(f"Pred: {data['Predicted FI Category']}")
                    st.caption(f"Actual: {data['Actual FI Category']}")

def calculator_knn_section():
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">K-Nearest Neighbors Food Insecurity Rate Calculator</h2>
        </div>
        """, unsafe_allow_html=True)
    st.write("Enter county and state to get predicted FI category and range.")
    # Dropdowns
    data = pd.read_csv('./data/data.csv')
    states = data['State'].unique()
    state_name = st.selectbox("State", states)
    counties = data[data['State'] == state_name]['County'].unique()
    county_name = st.selectbox("County", counties)
        
    if st.button("Get FI Category"):
        r = predict_county(county_name, state_name)
        if r is not None:
            display_prediction_card(r)

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

def calculator_section():
    # Get Model Resources
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Neural Network Food Insecurity Rate Calculator</h2>
        </div>
        """, unsafe_allow_html=True)
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
                input_data['last_year_cluster'] = st.selectbox("Cluster Group", metadata['unique_values']['Cluster'])
            else:
                input_data['last_year_cluster'] = st.selectbox("Cluster Group", ['0', '1', '2', '3'], help="Cluster assignment based on poverty/employment features")

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
        col3.metric("F1-Score", f"{macro_f1:.4f}")

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

# def kmeans_page():
#     clustering_data = joblib.load(os.path.join(NN_ARTIFACTS_DIR, "clustering_model.save"))
#     clustered_df = pd.read_csv(os.path.join(NN_ARTIFACTS_DIR, "data_with_clusters.csv"))

#     if clustering_data is not None and clustered_df is not None:
#             st.markdown("---")
#             st.header("🎯 K-Means Cluster Analysis")

#             cluster_col1, cluster_col2 = st.columns([1, 2])

#             with cluster_col1:
#                 st.subheader("Clustering Info")
#                 st.metric("Optimal K (Clusters)", clustering_data['optimal_k'])
#                 st.metric("Silhouette Score", f"{clustering_data.get('silhouette_score', 'N/A'):.3f}"
#                          if 'silhouette_score' in clustering_data else "N/A")

#                 st.markdown("**Clustering Features:**")
#                 for feat in clustering_data['features']:
#                     st.text(f"• {feat}")

#                 # Cluster distribution
#                 cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
#                 st.markdown("**Cluster Distribution:**")
#                 for cluster_id, count in cluster_counts.items():
#                     st.text(f"Cluster {cluster_id}: {count} counties")

#             with cluster_col2:
#                 st.subheader("Cluster Visualization")

#                 # 3D scatter plot of clusters
#                 cluster_features = clustering_data['features']
#                 if len(cluster_features) >= 3:
#                     fig_3d = px.scatter_3d(
#                         clustered_df,
#                         x=cluster_features[0],
#                         y=cluster_features[1],
#                         z=cluster_features[2],
#                         color='Cluster',
#                         title='3D Cluster Visualization',
#                         labels={'Cluster': 'Cluster ID'},
#                         color_continuous_scale='Viridis'
#                     )
#                     fig_3d.update_layout(height=500)
#                     st.plotly_chart(fig_3d, use_container_width=True)

#             # Cluster statistics comparison
#             st.subheader("📊 Cluster Characteristics")

#             cluster_stats = clustered_df.groupby('Cluster')[clustering_data['features']].mean()

#             fig_cluster_bars = go.Figure()
#             for feature in clustering_data['features']:
#                 fig_cluster_bars.add_trace(go.Bar(
#                     name=feature,
#                     x=[f"Cluster {i}" for i in cluster_stats.index],
#                     y=cluster_stats[feature]
#                 ))

#             fig_cluster_bars.update_layout(
#                 title='Average Feature Values by Cluster',
#                 xaxis_title='Cluster',
#                 yaxis_title='Average Value',
#                 barmode='group',
#                 template='plotly_white',
#                 height=400
#             )
#             st.plotly_chart(fig_cluster_bars, use_container_width=True)

#             # Food Insecurity by Cluster
#             st.subheader("🍽️ Food Insecurity Rate by Cluster")
#             cluster_fi = clustered_df.groupby('Cluster')['Food_Insecurity_Rate'].agg(['mean', 'min', 'max', 'std'])

#             fig_fi_cluster = go.Figure()
#             fig_fi_cluster.add_trace(go.Bar(
#                 x=[f"Cluster {i}" for i in cluster_fi.index],
#                 y=cluster_fi['mean'] * 100,
#                 error_y=dict(type='data', array=cluster_fi['std'] * 100),
#                 marker_color='indianred',
#                 text=(cluster_fi['mean'] * 100).round(2),
#                 textposition='outside'
#             ))

#             fig_fi_cluster.update_layout(
#                 title='Average Food Insecurity Rate by Cluster (with std dev)',
#                 xaxis_title='Cluster',
#                 yaxis_title='Food Insecurity Rate (%)',
#                 template='plotly_white',
#                 height=400
#             )
#             st.plotly_chart(fig_fi_cluster, use_container_width=True)

# <<<<<<<<<<<<<<<<<<<<<<< K-Means Page <<<<<<<<<<<<<<<<<<<<<<<


# >>>>>>>>>>>>>>>>>>>>>> PCA Page >>>>>>>>>>>>>>>>>>>>>>>>>>

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
    left_pca, right_pca = st.columns(2)

    with left_pca:
        plt.figure(figsize=(8,5))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='x', linestyle='--')
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA - Explained Variance Ratio")
        # st.write("Scree Plot")
        st.pyplot(plt)
        st.info("Based on the scree plot, we will use 4 components.")

    
    # plt.suptitle("KMeans Clusters on PCA Components", y=1.02)
    # st.write("Pairplot of KMeans Clusters on PCA Components")
    with right_pca:
        image_path = os.path.join(ROOT_DIR, "models", "kmeans-pca", "artifacts", "kmeans-pca-clusters.png")
        image = Image.open(image_path)
        st.image(image)

    # pcaDF = pd.read_csv(os.path.join(ROOT_DIR, "data", "pcaDF.csv"))

    # # Visualize clusters in PCA space
    # sns.pairplot(pcaDF, vars=['PC1','PC2','PC3','PC4'], hue='cluster', palette='Set2')

    # # Pipeline with scaler + KMeans
    # feature_cols = numericCols
    # pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('kmeans', KMeans(n_clusters=2, init='k-means++', n_init=15, random_state=67))
    # ])
    # pipe.fit(subData[feature_cols])

    # # Extract labels and centroids
    # subData['cluster'] = pipe['kmeans'].labels_
    # centroids_scaled = pipe['kmeans'].cluster_centers_
    # centroids_original = pipe['scaler'].inverse_transform(centroids_scaled)

    # # Centroids DataFrame
    # pd.set_option('display.max_columns', None)
    # cent_df = pd.DataFrame(centroids_original, columns=feature_cols)
    # st.subheader("KMeans Cluster Centroids")
    # cent_df.to_csv(os.path.join(ROOT_DIR, "models", "kmeans-pca", "artifacts", "kmeans_centroids.csv"), index=False)
    cent_df = pd.read_csv(os.path.join(ROOT_DIR, "models", "kmeans-pca", "artifacts", "kmeans_centroids.csv"))

    st.write("Centroids DataFrame")
    st.write(cent_df.T)


    st.markdown("""
    ## Cluster Analysis

The two clusters represent distinct socioeconomic and food-security profiles across the units in the dataset. Cluster 0 is characterized by substantially lower absolute counts of food-insecure individuals and households, lower total population and household totals, and generally lower poverty-related burdens. For example, the centroid for Num_Food_Insecure_Persons (\~9,600), POP_BELOW_POVERTY (\~9,800), and HOUSEHOLDS_TOTAL (\~25,700) suggests smaller, less densely populated communities with relatively moderated levels of economic hardship. The food insecurity rate (\~0.145) and child food insecurity rate (\~0.219) are meaningful but not extreme, and median household income (\~$46,400) is modest yet notably higher than typical high-poverty geographies. Overall, Cluster 0 represents moderately food-insecure, lower-population regions with more stable economic indicators.

In contrast, Cluster 1 reflects a dramatically different profile, with an order-of-magnitude increase in population and economic strain. These areas exhibit extremely high counts of food-insecure persons (\~425,000), food-insecure children (\~150,000), and households receiving SNAP (\~121,000). Poverty and unemployment burdens are also substantially higher in absolute size, and median household income (\~$54,200) is slightly higher than Cluster 0 but does not compensate for the much larger populations living below poverty. Interestingly, the rates of food insecurity and child food insecurity are similar to those of Cluster 0, but the total scale of affected individuals is vastly larger. Thus, Cluster 1 captures high-population, high-need metropolitan or regional centers where structural poverty affects a far larger volume of residents, even when rate-based indicators appear comparable.

Taken together, these clusters differentiate not merely by intensity of food insecurity in percentage terms, but by structural magnitude—Cluster 0 represents smaller, moderately burdened communities, whereas Cluster 1 captures large-scale, high-need population centers where social assistance demand, economic vulnerability, and food insecurity exist at a substantially greater scale. This distinction is especially relevant for resource allocation: policies optimized for Cluster 1 must address volume and infrastructure capacity, whereas interventions for Cluster 0 may focus on rural access, localized service gaps, and targeted support.
    """)

# <<<<<<<<<<<<<<<<<<<<<<< PCA Page <<<<<<<<<<<<<<<<<<<<<<<



# >>>>>>>>>>>>>>>>>>>>>> KMeans Page >>>>>>>>>>>>>>>>>>>>>>>>>>

def kmeans_page():
    # plt.figure(figsize=(8,6))
    # plt.scatter(subData['Child_Food_Insecurity_Rate'], 
    #             subData['Food_Insecurity_Rate'], 
    #             c=subData['cluster'].astype(int), cmap='rainbow', alpha=0.2)
    # plt.xlabel("Child Food Insecurity Rate")
    # plt.ylabel("Food Insecurity Rate")
    # plt.title("KMeans Clusters (Original Features)")
    # st.pyplot(plt)
    left_kmeans, right_kmeans = st.columns([2, 3])

    with left_kmeans:
        image_path = os.path.join(ROOT_DIR, "models", "kmeans-pca", "artifacts", "kmeans_init.png")
        image = Image.open(image_path)
        st.image(image)

        image_path = os.path.join(ROOT_DIR, "models", "kmeans-pca", "artifacts", "kmeans-pca-clusters.png")
        image = Image.open(image_path)
        st.image(image)

    with right_kmeans:
        K_values = list(range(1,15))
        with open('models/kmeans-pca/artifacts/elbow.txt', 'r') as f:
            wcss = json.load(f)

        # wcss = []
        # for k in K_values:
        #     pipe.set_params(kmeans__n_clusters=k)
        #     pipe.fit(subData[feature_cols])
        #     wcss.append(pipe['kmeans'].inertia_)

        # with open('models/kmeans/artifacts/elbow.txt', 'w') as f:
        #     json.dump(wcss, f)

        fig = px.line(x=K_values, y=wcss, markers=True,
                    title="Elbow Plot",
                    labels={"x":"Number of Clusters", "y":"WCSS"})
        st.plotly_chart(fig)

        with open('models/kmeans-pca/artifacts/sil_scores.txt', 'r') as f:
            sil_scores = json.load(f)

        # sil_scores = []
        K_values_sil = list(range(2,15))
        # for k in K_values_sil:
        #     pipe.set_params(kmeans__n_clusters=k)
        #     pipe.fit(subData[feature_cols])
        #     labels = pipe['kmeans'].labels_
        #     sil_scores.append(silhouette_score(subData[feature_cols], labels))
        
        # with open('models/kmeans/artifacts/sil_scores.txt', 'w') as f:
        #     json.dump(sil_scores, f)

        fig = px.line(x=K_values_sil, y=sil_scores, markers=True,
                    title="Silhouette Scores",
                    labels={"x":"Number of Clusters", "y":"Silhouette Score"})
        st.plotly_chart(fig)

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