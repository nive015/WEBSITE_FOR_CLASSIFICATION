import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Set Page Config with a Default Icon
st.set_page_config(
    page_title="Classification App",
    page_icon="🤖",  # Emoji as fallback icon
    layout="wide",
    initial_sidebar_state="auto",
)

# Apply custom CSS styles
st.markdown("""
    <style>
        .main {
            background-color: black; 
           /* bg for main area*/
        }
       .st-emotion-cache-1gv3huu { 
            background-color: red;
            color: #FF0000; /*sidebar bg*/
        }
        .st-emotion-cache-12fmjuu ezrtsby2{
            background-color: red; /*deploy thing bg*/
        }
       .st-emotion-cache-6qob1r eczjsme11{
            color: white; /*sidebar-content*/
        }
        .h2tags {
            color: #00FFFF;
            font-weight: bold;
        }
        .h3tags {
            color: white;
            font-size: +2;
            text-align: center;
        }
        ."st-emotion-cache-j7qwjs eczjsme15{
            font-color:white;
        }
    </style>
    """, unsafe_allow_html=True)

# Main title
st.markdown('<h1 style="color:red;font-family:Algerian;">Classification Algorithms</h1><hr>', unsafe_allow_html=True)

# Sidebar Image Handling
image_url = "https://raw.githubusercontent.com/nive015/WEBSITE_FOR_CLASSIFICATION/main/streamlit_classification/robo.jfif"

try:
    response = requests.get(image_url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    st.sidebar.image(image, use_column_width=True)
except Exception as e:
    st.sidebar.error(f"Image failed to load: {e}")

# Content
st.markdown("""
    <h2 class="h2tags">What is Classification?</h2><br><br>
    <h3 class="h3tags">Classification is a type of supervised learning in machine learning where the model learns to categorize data points into predefined classes.</h3>
    <h2 class="h2tags">Types of Classification Algorithms</h2>
    <h3 class="h3tags">
       <div style="text-align:left;"> 1. Decision Tree:</div><br><div style="text-align:right;"> A Decision Tree Classifier is a supervised machine learning algorithm...</div><br><br>
       <div style="text-align:left;"> 2. Random Forest:</div><br><div style="text-align:right;"> A Random Forest Classifier is an ensemble learning method...</div><br><br>
       <div style="text-align:left;"> 3. K-Nearest Neighbors (KNN):</div><br><div style="text-align:right;"> K-Nearest Neighbors (KNN) is a straightforward and effective classification algorithm...</div><br><br>
       <div style="text-align:left;"> 4. Logistic Regression:</div><br> <div style="text-align:right;">Logistic Regression is a widely used and simple classification algorithm...</div><br><br>
       <div style="text-align:left;"> 5. Support Vector Machine (SVM):</div> <br><div style="text-align:right;">Support Vector Machines (SVMs) are powerful classifiers...</div><br><br>
       <div style="text-align:left;"> 6. Naive Bayes:</div><br><div style="text-align:right;">Naive Bayes is a probabilistic classifier based on Bayes' Theorem...</div><br>
    </h3>
    """, unsafe_allow_html=True)
