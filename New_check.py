import streamlit as st
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: API Input and Fetching Data
def fetch_data():
    st.title("API Data Fetching and Machine Learning")

    # Get user inputs for API
    url = st.text_input("Enter API URL", "https://covid-193.p.rapidapi.com/statistics")
    query_string = st.text_area("Enter query parameters (optional)", "{}")
    headers_input = st.text_area("Enter headers (optional)", "{}")

    if st.button("Fetch Data"):
        try:
            headers = eval(headers_input) if headers_input else {}
            params = eval(query_string) if query_string else {}
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = pd.DataFrame(response.json()["response"])
                st.success("Data fetched successfully!")
                st.dataframe(data.head())  # Display the first few rows of data

                # Store data in session state
                st.session_state.data = data
                return data
            else:
                st.error("Error fetching data from API.")
        except Exception as e:
            st.error(f"Error occurred: {e}")
    return None

# Step 2: Data Preprocessing and Analysis
def data_preprocessing():
    if 'data' not in st.session_state:
        st.warning("No data available. Please fetch data first.")
        return None

    data = st.session_state.data
    st.subheader("Data Preprocessing")

    if st.checkbox("Show basic statistics"):
        st.write(data.describe())

    if st.checkbox("Handle missing values"):
        missing_value_method = st.radio("Select method to handle missing values", ('None', 'Fill with mean', 'Drop rows with missing values'))
        if missing_value_method == 'Fill with mean':
            data.fillna(data.mean(), inplace=True)
            st.success("Filled missing values with mean.")
        elif missing_value_method == 'Drop rows with missing values':
            data.dropna(inplace=True)
            st.success("Dropped rows with missing values.")

    if st.checkbox("Visualize pairplot"):
        st.write(sns.pairplot(data))
        st.pyplot()  # Render the Seaborn pairplot

    # Save preprocessed data in session state
    st.session_state.preprocessed_data = data
    return data

# Step 3: Model Training
def model_training():
    if 'preprocessed_data' not in st.session_state:
        st.warning("No preprocessed data available. Please preprocess the data first.")
        return None

    data = st.session_state.preprocessed_data
    st.subheader("Model Training")

    # Assuming 'target' is the column to predict
    target = st.selectbox("Select the target variable", data.columns)

    if st.button("Train Model"):
        X = data.drop(target, axis=1)
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Training a Random Forest Classifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Display confusion matrix
        st.write("Confusion Matrix:")
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
        st.pyplot()

        # Save the model in session state
        st.session_state.model = model
        return model

# Step 4: Making Predictions (optional step)
def model_prediction():
    if 'model' not in st.session_state:
        st.warning("No trained model available. Please train the model first.")
        return None

    st.subheader("Model Prediction")
    model = st.session_state.model

    # Allow user to input new data for prediction
    input_data = st.text_area("Enter new data for prediction (in JSON format)", "{}")
    if st.button("Predict"):
        try:
            new_data = pd.DataFrame([eval(input_data)])
            prediction = model.predict(new_data)
            st.write(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Error occurred during prediction: {e}")

# Streamlit App Flow
def main():
    # Initialize session state if not present
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose mode", ["Fetch Data", "Preprocess Data", "Train Model", "Make Prediction"])

    if app_mode == "Fetch Data":
        fetch_data()
    elif app_mode == "Preprocess Data":
        data_preprocessing()
    elif app_mode == "Train Model":
        model_training()
    elif app_mode == "Make Prediction":
        model_prediction()

if __name__ == '__main__':
    main()
