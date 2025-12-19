import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# --- CONFIGURATION ---
INPUT_CSV = 'mbpp_dt_train.csv'
MODEL_OUTPUT_FILE = 'llm_router_model_mbpp.pkl'
# ---------------------

def train_model():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print("Loading dataset...")
    df = pd.read_csv(INPUT_CSV)

    # Features must match the extraction order in your main script
    feature_cols = [
        'current_model_size', 
        'mean_confidence', 
        'bottom_window_confidence', 
        'tail_confidence', 
        'least_grouped_confidence'
    ]
    
    X = df[feature_cols]
    y = df['target_class'] # 0 = 4B, 1 = 32B

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Classifier
    # Restricting max_depth prevents memorizing the small dataset
    clf = DecisionTreeClassifier(
        max_depth=5, 
        min_samples_leaf=3,
        class_weight='balanced', 
        random_state=42,
        criterion='gini'
    )

    clf.fit(X_train, y_train)

    # Evaluate
    print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.2%}")
    print(classification_report(y_test, clf.predict(X_test), target_names=['Use 4B', 'Use 32B']))

    # Save
    joblib.dump({'model': clf, 'feature_names': feature_cols}, MODEL_OUTPUT_FILE)
    print(f"✅ Model saved to {MODEL_OUTPUT_FILE}")

if __name__ == "__main__":
    train_model()
