import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os

def train_model(data_path="data/ma_data.csv", model_output="model/ma_model.pkl"):
    # Load dataset
    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    # Features and Target
    X = df.drop("success", axis=1)
    y = df["success"]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train model
    model = XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("✅ Classification Report:\n", classification_report(y_test, y_pred))
    print("✅ ROC AUC Score:", roc_auc_score(y_test, y_proba))

    # Save the model
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(model, model_output)
    print(f"✅ Model saved to {model_output}")

if __name__ == "__main__":
    train_model()
