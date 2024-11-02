import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the training and test datasets
train_file = r'C:\Users\Dell\Desktop\titanic_project\train.json'
test_file = r'C:\Users\Dell\Desktop\titanic_project\test.json'

# Load the JSON data
with open(train_file) as f:
    train_data = json.load(f)
    
with open(test_file) as f:
    test_data = json.load(f)

# Convert to pandas DataFrame for easier manipulation
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Inspect the data
print(train_df.head())
print(test_df.head())

# Combine ingredients into a single string for each recipe
train_df['ingredients_text'] = train_df['ingredients'].apply(lambda x: ' '.join(x))
test_df['ingredients_text'] = test_df['ingredients'].apply(lambda x: ' '.join(x))

# Define features (ingredients) and target (cuisine)
X = train_df['ingredients_text']
y = train_df['cuisine']

# Split the training data for validation (optional)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Use TF-IDF Vectorization to convert text into numerical features
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(test_df['ingredients_text'])

# Build the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Validate the model
y_pred_val = model.predict(X_val_tfidf)
print(f'Validation Accuracy: {accuracy_score(y_val, y_pred_val)}')
print(classification_report(y_val, y_pred_val))

# Predict on the test set
test_predictions = model.predict(X_test_tfidf)

# Prepare the submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id'], 'cuisine': test_predictions})

# Save the predictions to a CSV file
submission_df.to_csv('cuisine_predictions.csv', index=False)
print('Predictions saved to cuisine_predictions.csv')
