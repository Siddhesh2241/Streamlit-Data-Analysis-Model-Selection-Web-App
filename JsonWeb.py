from Library import *
from Load_data import *
import io
from joblib import parallel_backend

st.set_page_config(page_title="Web Project", layout="centered",page_icon="📰")

def Analyse_data(data, file_type):

    # Always display the latest dataframe at the top
    st.subheader("Current Dataset View (First 5 Rows)")
    st.write(data.head())  # Display current dataframe after every change

    st.subheader("Columns in dataset:")
    st.write(data.columns.tolist())

    st.subheader("Numerical columns in dataset:")
    st.write([col for col in data.columns if data[col].dtypes != "object"])

    st.subheader("Categorical columns in dataset:")
    st.write([col for col in data.columns if data[col].dtypes == "object"])

    st.subheader("Information of the CSV file:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Statistical data of the dataset:")
    st.write(data.describe())

    if file_type != "json":
        st.subheader("Check for duplicates in the dataset:")
        st.write(data.duplicated().sum())

        st.subheader("Unique values in the dataset:")
        st.write(data.nunique())
    else:
        pass

    # Check for null values
    st.subheader("Check for null values in the dataset:")
    st.write(data.isnull().sum())
    
    return data

def Preprocess_data(data):
    if data.isnull().sum().sum() > 0:
        num_columns = st.number_input("How many columns would you like to work with?", min_value=1, max_value=len(data.columns), value=1, step=1)
        
        for i in range(int(num_columns)):
            st.write(f"### Operation for Column {i + 1}")
            obj = st.selectbox("Select the method", options=["Drop_column", "mean", "mode", "median"], key=f"method_{i}")

            if obj == "Drop_column":
                st.subheader("Dropping unnecessary columns")
                columns_to_drop = st.multiselect("Select column", data.columns.tolist(), key=f"drop_{i}")
                
                if st.button("Submit and drop column", key=f"drop_button_{i}"):
                    if columns_to_drop:
                        data = data.drop(columns=columns_to_drop)
                        st.session_state.df = data  # Update session state with dropped columns
                        st.write(f"### DataFrame after dropping columns: {columns_to_drop}")
                        st.write(data)
                    else:
                        st.write("No columns dropped.")

            elif obj == "mean":
                st.subheader("Fill missing values with mean of column")
                columns_to_mean = st.multiselect("Select column for mean", data.columns.tolist(), key=f"mean_{i}")
                
                if st.button("Submit and fill columns with mean", key=f"mean_button_{i}"):
                    if columns_to_mean:
                        for col in columns_to_mean:
                            data[col].fillna(data[col].mean(), inplace=True)
                        st.session_state.df = data  # Update session state after filling
                        st.write(f"### DataFrame after filling missing values with mean: {columns_to_mean}")
                        st.write(data)
                    else:
                        st.write("No columns filled.")
            
            elif(obj=="mode"):

                st.subheader("Fill missing values with mode of column")
                columns_to_mode = st.multiselect("Select column for mode ",data.columns.tolist())
                
                if st.button("Submit and fill columns", key=f"submit_mode_{i}"):
                    if columns_to_mode:
                       for col in columns_to_mode:
                         data[col].fillna(data[col].mode()[0], inplace=True)
                         st.session_state.df = data
                       st.write(f"### DataFrame after Filling Missing values : {columns_to_mode}")
                       st.write(data)
                    else:
                      st.write("No columns Fill.")
            
            elif(obj=="median"):

                st.subheader("Fill missing values with median of column")
                columns_to_median = st.multiselect("Select column for median",data.columns.tolist())
                
                if st.button("Submit and fill columns", key=f"submit_mode_{i}"):
                    if columns_to_median:
                       for col in columns_to_median:
                         data[col].fillna(data[col].median(), inplace=True)
                         st.session_state.df = data
                       st.write(f"### DataFrame after Filling Missing values : {columns_to_median}")
                       st.write(data)
                    else:
                      st.write("No columns Fill.")

    else:
        st.write("No missing values found in the dataset.")

    # Convert float to int
    if data.select_dtypes(include='float').shape[1] > 0:
        st.subheader("Convert Float Columns to Integer")

        input_data = st.selectbox("Convert float columns?", options=["yes", "no"], key="convert_float")

        if input_data == "yes":
            float_columns = data.select_dtypes(include='float').columns.tolist()
            columns_to_convert = st.multiselect("Select the float columns to convert", float_columns, key="float_convert")

            if st.button("Convert selected columns", key="convert_button"):
                if columns_to_convert:
                    data[columns_to_convert] = data[columns_to_convert].astype(int)
                    st.session_state.df = data  # Update session state after conversion
                    st.write(f"### DataFrame after converting the following columns to integers: {columns_to_convert}")
                    st.write(data)
                else:
                    st.write("No columns selected for conversion.")
        else:
            st.write("No conversion selected.")
    
    drop = st.selectbox("Can you drop some Columns?",options=["No","Yes"])

    if drop is not None:
        if drop == "Yes":
                st.subheader("Dropping unnecessary columns")
                columns_to_drop = st.multiselect("Select column", data.columns.tolist())
                
                if st.button("Submit and drop column",key="drop_columns"):
                    if columns_to_drop:
                        data = data.drop(columns=columns_to_drop)
                        st.session_state.df = data  # Update session state with dropped columns
                        st.write(f"### DataFrame after dropping columns: {columns_to_drop}")
                        st.write(data)
                    else:
                        st.write("No columns dropped.")
        else:
            pass
    
    st.subheader("Combine columns into a single string ")

    combine = st.multiselect("plz select the Column",options=data.columns.tolist())

    if combine is not None:
        if st.button("Combine columns",key="combine_columns"):
            def combine_ingredients(ingredient_list):
              return " ".join(map(str, ingredient_list))
            
            data[combine] = data[combine].apply(combine_ingredients)
            st.write("Ingredients combined into text format:")
            st.write(data.head())

    else:
        st.write("Plz select the columns")    

    # Always return the latest data to maintain state
    return data


def Encode_data(data):
    # Check if there are any categorical columns to encode
    if data.select_dtypes(include='object').shape[1] > 0:
        st.subheader("Encoding Ingredients with TF-IDF Vectorizer")
        
        # Separate textual and non-textual categorical columns
        textual_columns = [col for col in data.columns if data[col].apply(lambda x: isinstance(x, str)).all()]
        categorical_columns = [col for col in data.columns if data[col].dtypes == 'object' ]
        
        # Allow user to select columns to encode
        columns_to_encode = st.multiselect("Select text column to encode with TF-IDF", textual_columns, key="encode_columns_unique")
        categorical_to_label_encode = st.multiselect("Select categorical columns to label encode", categorical_columns, key="label_encode_unique")

        # TF-IDF Vectorizer
        if st.button("Encode Selected Columns"):
            if columns_to_encode:
                tfidf = TfidfVectorizer(stop_words='english')
                
                # Only encode the selected text column (usually ingredients)
                
                data[columns_to_encode] = data[columns_to_encode].apply(lambda x: ' '.join(x))  # Ensure ingredients are a string

                with parallel_backend('threading', n_jobs=-1):
                  X_tfidf = tfidf.fit_transform(data[columns_to_encode])

                 # Store TF-IDF matrix in session state for further use in modeling
                  st.session_state['tfidf_matrix'] = X_tfidf

                  st.write(f"TF-IDF applied. Matrix shape: {X_tfidf.shape}")
            
            
             # Label Encoding for other categorical columns
            elif categorical_to_label_encode:
              encoder = LabelEncoder()
              for col in categorical_to_label_encode:
                data[col] = encoder.fit_transform(data[col])
            
              st.write(f"Label Encoding applied to columns: {categorical_to_label_encode}")
              st.write(data)
            

            else:
             st.write("No categorical columns selected for label encoding.")
    else:
        st.write("No categorical columns to encode.")

    return data


def ML_Algo(data):
    st.subheader("Preparing data")
    
    # Select feature and label columns
    Features = st.multiselect("Enter the Feature column name", options=data.columns)
    Label = st.selectbox("Enter the label column name", options=data.columns)

    if Label and Features:
        X = data[Features]
        y = data[Label]

        st.write(f"Selected Features: {Features}")
        st.write(f"Selected Label: {Label}")

        # Split the data into training and testing sets
        test_size = st.slider("Select test size (percentage)", min_value=0.1, max_value=0.5, step=0.1, value=0.2)
        Random = st.slider("Select random_state value", min_value=1, max_value=100, step=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=Random)
        
        # Check if user wants to apply TF-IDF transformation
        if st.checkbox("Encoding Text Data with TF-IDF", value=False):
            st.subheader("Encoding Ingredients with TF-IDF Vectorizer")

            # Check if a text column exists (e.g., 'ingredients_text') in the selected features
            text_columns = [col for col in data.columns if data[col].apply(lambda x: isinstance(x, str)).all()]
            
            if len(text_columns) > 0:
                tfidf = TfidfVectorizer(stop_words='english')

                # Apply TF-IDF only to the text columns
                X_train_text = tfidf.fit_transform(X_train[text_columns[0]])
                X_test_text = tfidf.transform(X_test[text_columns[0]])

                # If there are other non-text features, concatenate them with the TF-IDF features
                if len(Features) > 1:
                    # Drop the text columns from original features
                    X_train_other = X_train.drop(columns=text_columns).values
                    X_test_other = X_test.drop(columns=text_columns).values

                    # Concatenate TF-IDF encoded text with the other features
                    from scipy.sparse import hstack
                    X_train = hstack([X_train_text, X_train_other])
                    X_test = hstack([X_test_text, X_test_other])
                else:
                    X_train = X_train_text
                    X_test = X_test_text
            else:
                st.error("No textual columns selected for TF-IDF encoding.")
            
            st.write("Training data shape after TF-IDF:", X_train.shape)
            st.write("Testing data shape after TF-IDF:", X_test.shape)

        # Check the final shapes of X and y
        st.write("Final Training data shape:", X_train.shape)
        st.write("Final Testing data shape:", X_test.shape)

        if st.checkbox("Scale the feature data?", value=True):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            st.write("Feature data has been scaled using StandardScaler.") 
        
        st.subheader("Suggestion: ")
        if data[Label].nunique() <= 5:
            st.write("According to Analyse your data , use Classification algorithm")
        else:
            st.write("According to Analyse your data , use Regression algorithm")
        
        st.subheader("Select the type of Algorithm")
        Algo = st.selectbox("Select algorithm",options = ["Classification","Regression"])
        
        if Algo is not None:
    
            if Algo == "Classification":
              
              classifiers = {
                 "Decision Tree": DecisionTreeClassifier(),
                 "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
                 "Logistic Regression": LogisticRegression(max_iter=200),
                 "Support Vector Machine": SVC(probability=True),
                 "Random Forest": RandomForestClassifier(n_estimators=100),
                 "Voting Classifier": VotingClassifier(estimators=[
                        ("DT", DecisionTreeClassifier()),
                        ("KNN", KNeighborsClassifier(n_neighbors=5)),
                        ("LR", LogisticRegression(max_iter=200)),
                        ("SVM", SVC(probability=True)),
                        ("RFC", RandomForestClassifier(n_estimators=100)),
                        ("Bagg_clf",BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)),
                        ("Boost_clf",AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, algorithm="SAMME", random_state=42))
                          ], voting="soft"),
                        "Bagging Classifier": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42),
                         "Boosting Classifier": AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, algorithm="SAMME", random_state=42)
                         }
              
              if st.checkbox("Do you want to compare multiple algorithms?"):

                selected_algorithms = []

                selected_algorithms = st.multiselect("Select Algorithms for Comparison", list(classifiers.keys()))

                if len(selected_algorithms) > 0:
                  # Number of subplots (1 plot per algorithm, each containing a bar for training and testing accuracy)
                  n_algorithms = len(selected_algorithms)
                  
                  rows = (n_algorithms - 1) // 3 + 1
                  cols = min(n_algorithms, 3)

                  fig, axes = plt.subplots(rows, cols, figsize =(cols * 6, rows * 6))

                  # If only one algorithm is selected, 'axes' is not an array, so handle that case
                  if n_algorithms == 1:
                    axes = [axes]
                  elif rows > 1 or cols > 1:
                    axes = axes.flatten()

                  for idx, algo_name in enumerate(selected_algorithms):
                    clf = classifiers[algo_name]
                    clf.fit(X_train, y_train)
            
                    # Get predictions
                    train_pred = clf.predict(X_train)
                    test_pred = clf.predict(X_test)
            
                    # Calculate accuracies
                    train_accuracy = accuracy_score(y_train, train_pred)
                    test_accuracy = accuracy_score(y_test, test_pred)

                    # Accuracy values
                    accuracies = {
                      "Training Accuracy": train_accuracy,
                      "Testing Accuracy": test_accuracy
                        }

                    # Create a bar plot for both training and testing accuracy on the same x-axis
                    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), ax=axes[idx])
                    axes[idx].set_title(f"{algo_name} Accuracy")
                    axes[idx].set_ylim(0, 1)  # Set y-axis limit for consistent scaling across all plots

                    # Add labels above the bars
                    for p in axes[idx].patches:
                      axes[idx].annotate(f'{p.get_height():.2f}', 
                                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                                   ha='center', va='baseline', fontsize=12, color='black', 
                                   xytext=(0, 5), textcoords='offset points')

                  plt.tight_layout()
                  st.pyplot(fig)

              model_name = st.selectbox("Select Classifier", list(classifiers.keys()))
              model = classifiers[model_name]

              if st.button("run Algo"):

               # Training and predictions
               if model_name in ["K-Nearest Neighbors", "Logistic Regression", "Support Vector Machine", "Voting Classifier","Bagging Classifier","Boosting Classifier"]:
                  model.fit(X_train, y_train)
                  train_pred = model.predict(X_train)
                  test_pred = model.predict(X_test)
               else:
                  model.fit(X_train, y_train)
                  train_pred = model.predict(X_train)
                  test_pred = model.predict(X_test)

                # Display accuracy scores
               st.write(f"Training Accuracy of {model_name} model:", accuracy_score(train_pred, y_train))
               st.write(f"Testing Accuracy of {model_name} model:", accuracy_score(test_pred, y_test))

               # Confusion matrix plot
               plt.figure(figsize=(8, 6))
               sns.heatmap(confusion_matrix(y_test, test_pred), annot=True, cmap="Blues")
               plt.xlabel("Actual values")
               plt.ylabel("Predicted values")
               plt.title(f"Confusion Matrix of {model_name} model")
               st.pyplot(plt)


def main():   
    st.title("Streamlit Web Project")

    df, file_type = Load_data()
    if df is None:
        st.error("Error: Failed to load the dataset. Please check the file.")
        return

    st.session_state.df = df if 'df' not in st.session_state else st.session_state.df
    updated_df = Analyse_data(st.session_state.df, file_type)
    preprocessed_df = Preprocess_data(updated_df)
    encoded_df = Encode_data(preprocessed_df)
    ML_Algo(encoded_df)


if __name__ == "__main__":
    main()