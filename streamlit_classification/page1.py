import streamlit as st
from PIL import Image

# Set page configuration
st.set_page_config(
    page_icon=Image.open("robo.jfif"),
    layout="wide",
    initial_sidebar_state="auto"
)

# Apply custom CSS styles
st.markdown("""
    <style>
        .main {
            background-color: black; /* Main background */
        }
        .st-emotion-cache-1gv3huu { 
            background-color: red !important;
        }
        .st-emotion-cache-6qob1r {
            color: white !important;
        }
        .h2tags {
            color: #00FFFF;
            font-weight: bold;
        }
        .h3tags {
            color: white;
            font-size: 20px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

# Main title
st.markdown('<h1 style="color:red;font-family:Algerian;font-size:24px;">Classification Algorithms</h1><hr>', unsafe_allow_html=True)

# Content
st.markdown("""
    <h2 class="h2tags">What is Classification?</h2><br><br>
    <h3 class="h3tags">Classification is a type of supervised learning in machine learning where the model learns to categorize data points into predefined classes.<br>
    The goal is to predict the category or class label of new data points based on the model's training on labeled data.</h3>

    <h2 class="h2tags">Types of Classification Algorithms</h2>
    <h3 class="h3tags">
        <div style="text-align:left;"> 1. Decision Tree:</div><br>
        <div style="text-align:right;">A Decision Tree Classifier is a supervised machine learning algorithm used for classification tasks. It works by recursively splitting the dataset into subsets based on input features.</div><br><br>

        <div style="text-align:left;"> 2. Random Forest:</div><br>
        <div style="text-align:right;">A Random Forest Classifier is an ensemble learning method that combines multiple decision trees to improve accuracy.</div><br><br>

        <div style="text-align:left;"> 3. K-Nearest Neighbors (KNN):</div><br>
        <div style="text-align:right;">KNN is a simple and effective classification algorithm, particularly useful when data is small and low-dimensional.</div><br><br>

        <div style="text-align:left;"> 4. Logistic Regression:</div><br> 
        <div style="text-align:right;">Logistic Regression is a widely used binary classification algorithm providing probabilistic predictions.</div><br><br>

        <div style="text-align:left;"> 5. Support Vector Machine (SVM):</div> <br>
        <div style="text-align:right;">SVMs are powerful classifiers that maximize the margin between classes using different kernels.</div><br><br>

        <div style="text-align:left;"> 6. Naive Bayes:</div><br>
        <div style="text-align:right;">Naive Bayes is a probabilistic classifier effective for large datasets, particularly in text classification.</div><br>
    </h3>
    """, unsafe_allow_html=True)

# Sidebar image
image_path = "robo.jfif"
try:
    image = Image.open(image_path)
    st.sidebar.image(image, caption="", use_column_width=True)
except Exception as e:
    st.sidebar.error(f"Error loading image: {e}")
