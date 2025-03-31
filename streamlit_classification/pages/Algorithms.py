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
import requests
from io import BytesIO

st.set_page_config(
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
    <style>
        .main {
            background-color: black;
            color:white;
        }
       .st-emotion-cache-1gv3huu { 
            background-color: red;
            color: #FF0000;
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
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 style="color:red;font-family:Algerian;font-size:+2;">Classification Algorithms</h1><hr>', unsafe_allow_html=True)

# Load image from URL
image_url = "https://raw.githubusercontent.com/nive015/WEBSITE_FOR_CLASSIFICATION/main/streamlit_classification/robo.jfif"
try:
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    st.sidebar.image(image, caption="", use_column_width=True)
except Exception as e:
    st.sidebar.error(f"Error loading image: {e}")

# File uploader
st.write("Accepted File Formats: csv, xlsx.")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = None

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or XLSX file.")

        numeric_cols = df.select_dtypes(include=['number']).columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns

        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        for column in non_numeric_cols:
            df[column] = df[column].fillna(df[column].mode()[0])

        label_encoders = {}
        for column in non_numeric_cols:
            try:
                label_encoders[column] = LabelEncoder()
                df[column] = label_encoders[column].fit_transform(df[column])
            except Exception as e:
                st.error(f"Error encoding {column}: {e}")

        st.markdown("""<h2 class="h2tags">Data After Preprocessing</h2>""", unsafe_allow_html=True)
        st.write(df)

        st.markdown("""<h2 class="h2tags">Select the target Column</h2>""", unsafe_allow_html=True)
        target_column = st.selectbox("", df.columns)
        X = df.drop(columns=target_column)
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        st.markdown("""<h2 class="h2tags">Select a Model for Training and Evaluation</h2>""", unsafe_allow_html=True)
        options = ["Decision Tree", "Random Forest Tree", "K-Nearest Neighbor",
                   "Logistic Regression", "Support Vector Machine", "Naive Bayes"]

        selected_option = st.selectbox("", options)
        st.write(f"You selected: {selected_option}")

        model = None
        if selected_option == "Decision Tree":
            model = DecisionTreeClassifier()
        elif selected_option == "Random Forest Tree":
            model = RandomForestClassifier(n_estimators=100)
        elif selected_option == "K-Nearest Neighbor":
            k = st.slider("Select the value of k:", min_value=1, max_value=20, value=5)
            model = KNeighborsClassifier(n_neighbors=k)
        elif selected_option == "Logistic Regression":
            model = LogisticRegression()
        elif selected_option == "Support Vector Machine":
            model = SVC(kernel='linear')
        elif selected_option == "Naive Bayes":
            model = GaussianNB()

        if model is not None:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.2f}")
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("Classification Report:")
            st.write(classification_report(y_test, y_pred))

            st.markdown("""<h2 class="h2tags">Select Visualization Graph</h2>""", unsafe_allow_html=True)

            plot_option = st.selectbox("", ["None", "Scatter Plot", "Bar Chart", "Line Plot", "Histogram"])

            if plot_option == "Scatter Plot":
                if X.shape[1] >= 2:
                    x_feature = st.selectbox("Select x-axis:", X.columns)
                    y_feature = st.selectbox("Select y-axis:", X.columns)
                    if x_feature and y_feature:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(x=X[x_feature], y=X[y_feature], hue=y, palette='viridis', ax=ax)
                        ax.set_xlabel(x_feature)
                        ax.set_ylabel(y_feature)
                        ax.set_title('Scatter Plot of Selected Features')
                        ax.legend(title='Target')
                        st.pyplot(fig)
                else:
                    st.warning("Scatter plot requires at least 2 features.")

            elif plot_option == "Bar Chart":
                bar_column = st.selectbox("Select column for Bar Chart:", df.columns)
                fig, ax = plt.subplots(figsize=(10, 6))
                df[bar_column].value_counts().plot(kind='bar', ax=ax)
                ax.set_xlabel(bar_column)
                ax.set_ylabel('Count')
                ax.set_title(f'Bar Chart of {bar_column}')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                st.pyplot(fig)

            elif plot_option == "Line Plot":
                x_column = st.selectbox("Select x-axis for Line Plot:", df.columns)
                y_column = st.selectbox("Select y-axis for Line Plot:", df.columns)
                if x_column and y_column:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df.plot(kind='line', x=x_column, y=y_column, ax=ax)
                    ax.set_xlabel(x_column)
                    ax.set_ylabel(y_column)
                    ax.set_title(f'Line Plot of {y_column} vs {x_column}')
                    st.pyplot(fig)

            elif plot_option == "Histogram":
                hist_column = st.selectbox("Select column for Histogram:", df.columns)
                fig, ax = plt.subplots(figsize=(10, 6))
                df[hist_column].plot(kind='hist', bins=30, ax=ax)
                ax.set_xlabel(hist_column)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Histogram of {hist_column}')
                st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.warning("Upload a file in CSV or XLSX format.")
