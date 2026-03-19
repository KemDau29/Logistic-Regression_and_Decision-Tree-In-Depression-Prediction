import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle as pk
import pandas as pd
from source.data_preprocessing import load_data, preprocess_data
from source.model_decision_tree import DecisionTree
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report


# Load & Preprocess
df = load_data()
X_train, X_test, y_train, y_test, encoders = preprocess_data(df)

# Train model
model = DecisionTree(max_depth=7)
model.fit(X_train.values, y_train.values)

# Evaluate
y_pred = model.predict(X_test.values)

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Save artifact
artifact = {
    "model_DT": model,
    "feature_cols": X_train.columns.tolist(),
    "encoders": encoders
}
with open("decision_tree.pkl", "wb") as f:
    pk.dump(artifact, f)

print("Model saved as 'decision_tree.pkl'")
