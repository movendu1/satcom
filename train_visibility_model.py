import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("satcom_dataset.csv")

# Encode orbit type
le = LabelEncoder()
df["orbit_type_enc"] = le.fit_transform(df["orbit_type"])

# Features & label
X = df[
    [
        "orbit_type_enc",
        "elevation",
        "distance_km",
        "gs_lat",
        "gs_lon"
    ]
]
y = df["good_link"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
import joblib

joblib.dump(model, "visibility_model.pkl")
print("✅ Model saved as visibility_model.pkl")
