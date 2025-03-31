# WEBSITE_FOR_CLASSIFICATION

DESCRIPTION:
  ABOUT THE WEBSITE:
    THIS WEBSITE MAINLY AIMS ON PERFORMING CLASSIFICATION FOR ANY GIVEN DATASET. THE USER WILL SIMPLY HAVE TO UPLOAD A DATASET WHICH IS IN .CSV OR .XLSX FORMAT. FURTHER, THE DATA=PREPROESSING, TRAINING USING MULTIPLE MODELS AND VISUALISATION WILL BE ENABLED.
 



 **1. Streamlit Page Setup**
- `st.set_page_config(...)` sets the page icon (`robo.jfif`), layout, and sidebar state.
- **Custom CSS styling**:
  - Changes the background color of the main area to black.
  - Customizes the sidebar (`red` background).
  - Styles `h2` and `h3` headings with different colors and fonts.


## **2. Sidebar and Introduction**
- **Sidebar image**: Displays `robo.jfif` as a logo.
- **Main Title**: Displays `"Classification Algorithms"` in a red `Algerian` font.
- **Introduction to Classification**:
  - Provides a **textual description** of what classification is.
  - Lists **six common classification algorithms** with explanations.

---

## **3. File Upload and Data Handling**
- The app allows users to upload a CSV or XLSX file.
- Reads the file into a Pandas DataFrame.
- **Handles missing values**:
  - Numeric columns → filled with mean values.
  - Categorical columns → filled with mode (most frequent value).
- **Encodes categorical columns** using `LabelEncoder` so models can process them.

---

## **4. Selecting Target Column & Data Splitting**
- Users select the **target column** (dependent variable).
- The dataset is split into **features (X) and target (y)**.
- Splits the data into **training (70%) and testing (30%) sets** using `train_test_split()`.

---

## **5. Model Selection and Training**
Users select a classification algorithm:
- **Decision Tree** (`DecisionTreeClassifier()`)
- **Random Forest** (`RandomForestClassifier(n_estimators=100)`)
- **K-Nearest Neighbors** (`KNeighborsClassifier(n_neighbors=k)`) – slider to select `k`.
- **Logistic Regression** (`LogisticRegression()`)
- **Support Vector Machine (SVM)** (`SVC(kernel='linear')`)
- **Naive Bayes** (`GaussianNB()`)

**Training Process:**
1. The selected model is trained on `X_train, y_train`.
2. Predictions are made on `X_test`.
3. Model performance is evaluated using:
   - **Accuracy Score**
   - **Confusion Matrix**
   - **Classification Report** (precision, recall, F1-score)

---

## **6. Visualization Options**
Users can choose **different types of graphs**:
- **Decision Boundary**: Plots decision regions (only for datasets with 2 features).
- **Scatter Plot**: User selects `x` and `y` features, then plots points colored by class.
- **Bar Chart**: Shows frequency of values for a selected column.
- **Line Plot**: Plots one column against another.
- **Histogram**: Displays the distribution of a selected column.

---

## **7. Error Handling**
- **Try-except blocks** catch errors like:
  - Invalid file format.
  - Issues with data encoding.
  - Model training failures.

---

### **Overall Workflow**
1. User **uploads a dataset**.
2. The app **preprocesses data** (fills missing values, encodes labels).
3. User **selects a target column**.
4. Data is **split into train/test sets**.
5. User **chooses a classification algorithm**.
6. Model is **trained, tested, and evaluated**.
7. User can **visualize the results**.

This app provides an **interactive machine learning experience** for non-technical users, allowing them to train and compare classification models without writing code.

---

### **Possible Enhancements**
1. **More algorithms** (e.g., XGBoost, Neural Networks).
2. **Hyperparameter tuning** for better model performance.
3. **Feature selection** to remove irrelevant columns.
4. **Cross-validation** for more reliable accuracy estimation.

Would you like me to refine or add anything else? 😊

Here is the for the video recording of my project:

https://drive.google.com/file/d/1D2pqTlsk9waNGDOVQLXvqefqehK96IQK/view?usp=drive_link 

Here is the link for the streamlit app:

https://websiteforclassification-azcxj6kh42n7ab2gnr6bkq.streamlit.app/    
