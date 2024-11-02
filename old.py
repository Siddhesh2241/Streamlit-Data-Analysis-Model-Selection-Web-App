import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import sqlite3
import requests
import json

# Database connection functions
def create_connection(db_file):
    conn = sqlite3.connect(db_file)
    return conn

def create_table(conn):
    sql_create_table = """ CREATE TABLE IF NOT EXISTS diabetes_data (
                            id INTEGER PRIMARY KEY,
                            feature_1 REAL,
                            feature_2 REAL,
                            target INTEGER
                        ); """
    conn.execute(sql_create_table)

# Initialize database
database = "diabetes.db"
conn = create_connection(database)
create_table(conn)

# Streamlit app
st.title("Diabetes Prediction")

# Select data source
data_source = st.selectbox("Select Data Source", ["CSV", "SQL", "JSON", "API"])

if data_source == "CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())
        df.to_sql('diabetes_data', conn, if_exists='append', index=False)
        st.success("Data saved to database!")

elif data_source == "SQL":
    # Load data from SQL
    query = st.text_area("Enter SQL query to fetch data", "SELECT * FROM diabetes_data")
    if st.button("Fetch Data"):
        df = pd.read_sql_query(query, conn)
        st.write("Data Preview:", df.head())

elif data_source == "JSON":
    json_input = st.text_area("Paste JSON data here")
    if st.button("Load JSON"):
        df = pd.json_normalize(json.loads(json_input))
        st.write("Data Preview:", df.head())
        df.to_sql('diabetes_data', conn, if_exists='append', index=False)
        st.success("Data saved to database!")

elif data_source == "API":
    api_url = st.text_input("Enter API URL")
    if st.button("Fetch Data from API"):
        response = requests.get(api_url)
        if response.status_code == 200:
            df = pd.json_normalize(response.json())
            st.write("Data Preview:", df.head())
            df.to_sql('diabetes_data', conn, if_exists='append', index=False)
            st.success("Data saved to database!")
        else:
            st.error("Failed to fetch data from the API.")

# User input for prediction
st.subheader("User Input for Prediction")

def user_input_features():
    feature_1 = st.number_input("Feature 1")
    feature_2 = st.number_input("Feature 2")
    return np.array([[feature_1, feature_2]])

input_data = user_input_features()

# Model selection for classification
st.subheader("Model Selection")

model = st.selectbox("Select Model", ["Decision Tree", "KNN", "Logistic Regression", "SVM", "Random Forest"])

# Load dataset for training
data = pd.read_sql_query("SELECT * FROM diabetes_data", conn)
X = data.drop(columns=["target", "id"])  # assuming 'target' is the label column
y = data["target"]

# Train and evaluate models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model == "Decision Tree":
    clf = DecisionTreeClassifier()
elif model == "KNN":
    clf = KNeighborsClassifier()
elif model == "Logistic Regression":
    clf = LogisticRegression()
elif model == "SVM":
    clf = SVC()
elif model == "Random Forest":
    clf = RandomForestClassifier()

clf.fit(X_train, y_train)
predictions = clf.predict(input_data)
st.write("Predictions:", predictions)

# Display metrics
accuracy = accuracy_score(y_test, clf.predict(X_test))
confusion = confusion_matrix(y_test, clf.predict(X_test))

st.write("Accuracy:", accuracy)
st.write("Confusion Matrix:", confusion)

# Regression model integration
st.subheader("Regression Model")

def user_input_regression_features():
    feature_1 = st.number_input("Feature 1 for Regression")
    feature_2 = st.number_input("Feature 2 for Regression")
    return np.array([[feature_1, feature_2]])

reg_input_data = user_input_regression_features()

reg_model = st.selectbox("Select Regression Model", ["Linear Regression"])

if reg_model == "Linear Regression":
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)  # Assuming y is continuous for regression
    reg_prediction = lin_reg.predict(reg_input_data)
    st.write("Regression Prediction:", reg_prediction)

# Close database connection
conn.close()


''' def main():
    st.title("Streamlit Web Project")

    # Load the data
    df, file_type = Load_data()

    if df is None:
        st.error("Error: Failed to load the dataset. Please check the file.")
        return

    # Initialize session state to store dataframe
    if 'df' not in st.session_state :
        st.session_state.df = df
    
    if not isinstance(st.session_state.df, pd.DataFrame):
      st.error("Error: Data in session state is not a DataFrame.")
      return
    
    # Always use the session state dataframe to perform operations
    updated_df = Analyse_data(st.session_state.df, file_type)
    

    Preprocess = Preprocess_data(updated_df)
    
    Encode = Encode_data(Preprocess)

    Explore_data_analysis(Encode)
    
    Algo = ML_Algo(Encode)

    # Update session state with the most recent changes
    st.session_state.df = Algo

if __name__ == "__main__":
    main() 

'''

selected_classifiers = st.multiselect("Select Classifiers for Comparison", list(classifiers.keys()), default=list(classifiers.keys()))
                if st.button("Run and Compare Models"):
                    fig, axes = plt.subplots(1, len(selected_classifiers), figsize=(18, 6))

                    for i, model_name in enumerate(selected_classifiers):
                        model = classifiers[model_name]
                        model.fit(X_train, y_train)
                        test_pred = model.predict(X_test)

                        # Plot confusion matrix as a subplot for each classifier
                        sns.heatmap(confusion_matrix(y_test, test_pred), annot=True, cmap="Blues", ax=axes[i])
                        axes[i].set_title(f'{model_name}\nAccuracy: {accuracy_score(y_test, test_pred):.2f}')
                        axes[i].set_xlabel("Actual")
                        axes[i].set_ylabel("Predicted")
                    
                    st.pyplot(fig)

                    # You can also display comparison metrics in a table format if needed
                    st.subheader("Accuracy Comparison")
                    results = {}
                    for model_name in selected_classifiers:
                        model = classifiers[model_name]
                        model.fit(X_train, y_train)
                        test_pred = model.predict(X_test)
                        acc = accuracy_score(y_test, test_pred)
                        results[model_name] = acc

                    st.table(pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"]).sort_values(by="Accuracy", ascending=False))