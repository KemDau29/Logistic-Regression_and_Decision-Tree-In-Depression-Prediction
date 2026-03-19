import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle as pk 
from source.data_preprocessing import load_data, preprocess_data
from source.model_logistic_regression import LogisticRegressionScratch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load & Preprocess
df = load_data()
X_train, X_test, y_train, y_test, encoders = preprocess_data(df)

#List of the features before convert to arrays
feature_cols = X_train.columns.tolist()



# Convert pandas -> numpy arrays
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# Train model
model = LogisticRegressionScratch(learning_rate=0.01, n_iters=10000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\n Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Save model with pickle
artifact = {
    "model_LR": model,
    "feature_cols": feature_cols,
    "encoders": encoders
}

with open("logistic_regression.pkl", "wb") as f:
    pk.dump(artifact, f)
print("Model have been saved in 'logistic_regression.pkl'")