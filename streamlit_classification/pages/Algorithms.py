import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from PIL import Image
import os

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="ML Classification App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Styling
st.markdown("""
    <style>
        .main {
            background-color: black;
            color:white; /* Background for main area */
        }
        .st-emotion-cache-1gv3huu { 
            background-color: red;
        }
        .st-emotion-cache-12fmjuu {
            background-color: red;
        }
        .st-emotion-cache-6qob1r {
            color: white;
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
        .stCheckbox > label {
            color: #FF6347;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 style="color:red;font-family:Algerian;">Classification Algorithms</h1><hr>', unsafe_allow_html=True)

# Sidebar Image Handling
try:
    image_path = "robo.jfif"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.sidebar.image(image, caption="Classification Model", use_column_width=True)
    else:
        st.sidebar.warning("⚠️ Image 'robo.jfif' not found. Please check the file path.")
except Exception as e:
    st.sidebar.error(f"Error loading image: {e}")

# File Upload Section
st.write("Accepted File Formats: CSV, XLSX.")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read the file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or XLSX file.")
            st.stop()

        # Handling missing values
        numeric_cols = df.select_dtypes(include=['number']).columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns

        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        for column in non_numeric_cols:
            df[column] = df[column].fillna(df[column].mode()[0])

        # Label Encoding for categorical columns
        label_encoders = {}
        for column in non_numeric_cols:
            try:
                label_encoders[column] = LabelEncoder()
                df[column] = label_encoders[column].fit_transform(df[column])
            except Exception as e:
                st.error(f"Error encoding {column}: {e}")

        # Display Preprocessed Data
        st.markdown('<h2 class="h2tags">Data After Preprocessing</h2>', unsafe_allow_html=True)
        st.write(df)

        # Selecting the Target Column
        st.markdown('<h2 class="h2tags">Select the Target Column</h2>', unsafe_allow_html=True)
        target_column = st.selectbox("Select Target Variable", df.columns)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Splitting Data into Train-Test Sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Selecting a Classification Model
        st.markdown('<h2 class="h2tags">Select a Model for Training</h2>', unsafe_allow_html=True)

        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "K-Nearest Neighbor": KNeighborsClassifier(),
            "Logistic Regression": LogisticRegression(),
            "Support Vector Machine": SVC(kernel='linear'),
            "Naive Bayes": GaussianNB()
        }

        selected_model = st.selectbox("Select Algorithm", list(models.keys()))
        model = models[selected_model]

        # Training the Model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Model Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Model Accuracy:** {accuracy:.2f}")
        st.write("**Confusion Matrix:**")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("**Classification Report:**")
        st.write(classification_report(y_test, y_pred))

        # Visualization Options
        st.markdown('<h2 class="h2tags">Select Visualization Graph</h2>', unsafe_allow_html=True)

        plot_option = st.selectbox("Choose Graph", ["None", "Scatter Plot", "Bar Chart", "Line Plot", "Histogram"])

        if plot_option == "Scatter Plot":
            if X.shape[1] >= 2:
                x_feature = st.selectbox("X-axis Feature", X.columns)
                y_feature = st.selectbox("Y-axis Feature", X.columns)
                if x_feature and y_feature:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=X[x_feature], y=X[y_feature], hue=y, palette='viridis', ax=ax)
                    ax.set_xlabel(x_feature)
                    ax.set_ylabel(y_feature)
                    ax.set_title('Scatter Plot of Selected Features')
                    st.pyplot(fig)
            else:
                st.warning("Scatter plot requires at least 2 features.")

        elif plot_option == "Bar Chart":
            bar_column = st.selectbox("Select Column", df.columns)
            fig, ax = plt.subplots(figsize=(10, 6))
            df[bar_column].value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel(bar_column)
            ax.set_ylabel('Count')
            ax.set_title(f'Bar Chart of {bar_column}')
            st.pyplot(fig)

        elif plot_option == "Line Plot":
            x_column = st.selectbox("X-axis", df.columns)
            y_column = st.selectbox("Y-axis", df.columns)
            if x_column and y_column:
                fig, ax = plt.subplots(figsize=(10, 6))
                df.plot(kind='line', x=x_column, y=y_column, ax=ax)
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_title(f'Line Plot of {y_column} vs {x_column}')
                st.pyplot(fig)

        elif plot_option == "Histogram":
            hist_column = st.selectbox("Select Column", df.columns)
            fig, ax = plt.subplots(figsize=(10, 6))
            df[hist_column].plot(kind='hist', bins=30, ax=ax)
            ax.set_xlabel(hist_column)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram of {hist_column}')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Upload a CSV or XLSX file to proceed!")
