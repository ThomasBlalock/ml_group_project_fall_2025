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

# """
# To run:
# streamlit run streamlit_app.py
# """

def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def key_visualizations():
    # Key Visualizations
    st.divider()
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Key Visualizations</h2>
        </div>
        """, unsafe_allow_html=True)
    st.bar_chart([6, 1, 1, 7, 3])


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
    key_visualizations()


def research_questions():
    # Research Questions
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Research Questions</h2>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""- What predictors best correlate food insecurity? <- figure this out agfter we make models (put little paragraph why its import and what data we need to can get us there)
- What model can best predict food insecurity?
- What are the most effective interventions to reduce food insecurity?
- Have the causes of food insecurity changed over time?""")


def models():
    # Models
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Machine Learning Models</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Buttons For Models
    root = '/home/blalo/uva/pred_modeling_i/ml_group_project_fall_2025/code/thomas/assets/'
    model_names = ["Linear Model", "KNN", "K-Means", "PCA", "Neural Network"]
    model_images = [
        os.path.join(root, "linear_model.jpg"),
        os.path.join(root, "knn_model.jpg"),
        os.path.join(root, "kmeans_model.png"),
        os.path.join(root, "pca_model.png"),
        os.path.join(root, "neural_network_model.jpg")
    ]

    selected_model = image_select(
        label="",
        images=model_images,
        captions=model_names,
        use_container_width=False,
        index=0,  # Default selected index (optional)
        return_value="original" # Returns the image path, we can also use "index"
    )
    if selected_model == model_images[0]:
        st.subheader("Linear Model Results")
        st.line_chart([1, 5, 2, 8])

    elif selected_model == model_images[1]:
        st.subheader("K-Nearest Neighbors Results")
        st.bar_chart([4, 2, 7, 5])

    elif selected_model == model_images[2]:
        st.subheader("K-Means Clustering Results")
        st.area_chart([9, 4, 5, 1])

    elif selected_model == model_images[3]:
        st.subheader("PCA Results")
        st.bar_chart([1, 1, 2, 3, 5])

    elif selected_model == model_images[4]:
        st.subheader("Neural Network Results")
        st.line_chart([8, 7, 6, 5, 4, 3])


def calculator(): # Home page
    # Interactive ML Models
    research_questions()
    st.divider()
    st.markdown("""
        <div style="padding:10px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Food Insecurity Calculator</h2>
        </div>
        """, unsafe_allow_html=True)
    st.write("We'll put something here where they input data and our models run in the background and give thema prediction.")



def home_page():
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
    c, ds, m = st.tabs(["Home", 
                        "Data Sources", 
                        "Models"], default="Home")

    with ds:
        data_sources()
    with m:
        models()
    with c:
        calculator()

home_page()