import joblib
import numpy as np
import tensorflow as tf
import os

pkl_path = r'test_records\test_8.pkl'

if not os.path.exists(pkl_path):
    print(" File not found!")
    exit()

print("Loading models...")

scaler = joblib.load("scaler.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
cnn_model = tf.keras.models.load_model(
    "cnn_transformer_model.h5",
    compile=False
)

print("Models loaded successfully!")

record = joblib.load(pkl_path)

X = np.array(record["features"]).reshape(1, -1)

X_scaled = scaler.transform(X)
X_cnn = X_scaled.reshape(1, X_scaled.shape[1], 1)


features = cnn_model.predict(X_cnn)
prediction = xgb_model.predict(features)[0]

all_tree_preds = []

for tree in xgb_model.get_booster().get_dump():
    all_tree_preds.append(prediction)

all_tree_preds = np.array(all_tree_preds)

confidence = 1 - (np.std(all_tree_preds) / (np.abs(prediction) + 1e-6))

confidence = float(np.clip(confidence, 0, 1))
print("\n===== RESULT =====")
print(f"Predicted Glucose Level: {prediction:.2f}")

if "glucose" in record:
    print(f"Actual Glucose Level: {record['glucose']:.2f}")

print(f"Confidence Score: {confidence:.2f}")
print("==================")