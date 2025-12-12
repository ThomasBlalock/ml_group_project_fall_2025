# Taking a Byte Out of Food Insecurity 🍽️

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://taking-a-byte-out-of-food-insecurity.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

**University of Virginia • Master of Data Science • Fall 2025**
*Predictive Modeling I Group Project*

### 🚀 **[View the Deployed Dashboard](https://taking-a-byte-out-of-food-insecurity.streamlit.app/)**

---

## 📖 Overview

**Forecasting Resilience: A Comparative Analysis of Predictive Models for County-Level Food Insecurity**

Food insecurity remains a pervasive challenge in the United States, deeply entrenched in local socioeconomic conditions. While the USDA subsidizes access, drivers of hunger vary significantly across geographies.

This project bridges the gap between national statistics and local realities. By combining **Feeding America's "Map the Meal Gap"** data with granular **US Census (ACS)** socioeconomic indicators, we trained and evaluated multiple machine learning architectures—from General Linear Models to Neural Networks—to model the landscape of American food insecurity.

### Key Research Questions
1.  **Model Efficacy:** Which architecture (GLM, KNN, NN) yields the lowest error rates?
2.  **Feature Importance:** Which socioeconomic variables are the strongest predictors of hunger?
3.  **Clustering:** Can U.S. counties be grouped into distinct "typologies" for tailored policy interventions?
4.  **Spatial Signal:** Is there a strong spatial autocorrelation to food insecurity?

---

## 📊 Data Sources

We utilize a dataset spanning **2011–2018**, merging "ground truth" hunger metrics with independent socioeconomic predictors.

| Source | Description | Key Variables |
| :--- | :--- | :--- |
| **US Census (ACS)** | Primary source for independent predictors capturing economic stability and structural vulnerability. | Median Income, Poverty Rate, Unemployment, SNAP Participation, Population Demographics. |
| **Map the Meal Gap** | Proprietary data from Feeding America serving as the target variable (ground truth). | Food Insecurity Rate, Cost Per Meal, Budget Shortfall, Child Food Insecurity. |

---

## 🧠 Modeling Approach

We implemented an interactive Streamlit dashboard allowing users to explore four distinct modeling approaches:

### 1. Linear Models (GLM)
* **Purpose:** Baseline prediction and feature interpretability.
* **Key Findings:** Established strong correlations between poverty rates/income and food insecurity, though struggled with complex non-linear relationships.

### 2. K-Nearest Neighbors (KNN)
* **Purpose:** Geographic and spatial analysis.
* **Key Findings:** Demonstrated strong spatial autocorrelation—counties act like their neighbors. Hunger is rarely an isolated local failure but a regional systemic issue.

### 3. Unsupervised Learning (K-Means & PCA)
* **Purpose:** Identifying county typologies.
* **Key Findings:** Identified a bifurcation in policy needs:
    * **Cluster A (Rural):** Requires targeted support to address localized service gaps and logistical isolation.
    * **Cluster B (Urban):** Requires volume and infrastructure capacity to handle high absolute numbers.

### 4. Neural Network (PyTorch)
* **Purpose:** Maximizing predictive power ($R^2$).
* **Architecture:** Multi-layer Perceptron (MLP) with Batch Normalization and Dropout layers.
* **Performance:** Achieved the highest $R^2$ and F-Score, effectively modeling complex non-linear interactions between variables.

---

## 💻 Installation & Usage

To run this project locally, follow these steps:

### Prerequisites
* Python 3.8+
* pip

### Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/ThomasBlalock/ml_group_project_fall_2025
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The app will automatically attempt to install missing packages (`streamlit`, `streamlit_image_select`, `base64`) if not found.*

3.  **Run the App**
    ```bash
    streamlit run streamlit_app.py
    ```

---

## 📂 Project Structure

```text
ML_GROUP_PROJECT_FALL_2025/
├── assets/                  # Static images for the UI (model diagrams)
├── data/                    # Processed datasets (CSV)
│   ├── data.csv             # Main dataset
│   ├── nn_df.csv            # Neural Net specific data
│   └── data_w_clusters.csv  # Data with K-Means labels
├── data_engineering/        # Jupyter notebooks for cleaning & merging
│   ├── get_census_data.py   # Census API script
│   └── MMG_merge_data.ipynb # Data merging logic
├── models/                  # Saved models and artifacts
│   ├── kmeans-pca/          # Clustering artifacts
│   ├── knn/                 # KNN confusion matrices & results
│   ├── linear/              # OLS results images
│   └── neural_net/          # PyTorch .pth models & scalers
├── streamlit_app.py         # Main Application Entry Point
└── requirements.txt         # Project dependencies
```

## 🏆 Conclusions & Future Work
Our analysis indicates that Neural Networks provide the most accurate forecasts, but interpretable models like K-Means offer actionable policy insights.

Policy Insight: We recommend lowering SNAP threshold requirements for high-risk areas identified in our analysis to cover working-poor households currently falling into the "aid gap."

Future Directions:

Temporal Validation: Test models against post-2018 data.

Explainable AI (XAI): Apply SHAP values to the Neural Network to make it "white-box."

Expanded Features: Incorporate transit access and grocery store density.

### 👥 Contributors
University of Virginia MSDS Team:

Muhammad Amjad

Reed Baumgardner

Thomas Blalock

Helen Corbat

Max Ellingsen

Harry Millspaugh

John Twomey
