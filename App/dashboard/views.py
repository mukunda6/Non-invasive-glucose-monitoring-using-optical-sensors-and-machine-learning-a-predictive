from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Prediction
from django.db.models import Count

import joblib
import numpy as np
import tensorflow as tf
import os

# Load models once (same as your script)
print("Loading models...")

scaler = joblib.load("model/scaler.pkl")
xgb_model = joblib.load("model/xgboost_model.pkl")
cnn_model = tf.keras.models.load_model(
    "model/cnn_transformer_model.h5",
    compile=False
)

print("Models loaded successfully!")

@login_required
def dashboard_view(request):
    return render(request, 'dashboard/dashboard.html')

@login_required
def predict_view(request):
    result = None
    actual = None
    confidence = None

    if request.method == "POST":
        uploaded_file = request.FILES.get("input_file")

        if not uploaded_file:
            result = "File not uploaded!"
        else:
            try:
                # ✅ SAME as: record = joblib.load(pkl_path)
                record = joblib.load(uploaded_file)

                # ✅ SAME
                X = np.array(record["features"]).reshape(1, -1)

                X_scaled = scaler.transform(X)
                X_cnn = X_scaled.reshape(1, X_scaled.shape[1], 1)

                features = cnn_model.predict(X_cnn)
                prediction = xgb_model.predict(features)[0]

                # ✅ SAME LOGIC (unchanged)
                all_tree_preds = []

                for tree in xgb_model.get_booster().get_dump():
                    all_tree_preds.append(prediction)

                all_tree_preds = np.array(all_tree_preds)

                confidence = 1 - (np.std(all_tree_preds) / (np.abs(prediction) + 1e-6))
                confidence = float(np.clip(confidence, 0, 1))

                # ✅ Prepare outputs (instead of print)
                result = f"{prediction:.2f}"

                if "glucose" in record:
                    actual = f"{record['glucose']:.2f}"

                # ✅ Save to DB
                Prediction.objects.create(
                    user=request.user,
                    input_file=uploaded_file,
                    predicted_class=result,
                    confidence=confidence
                )

            except Exception as e:
                result = f"Error: {str(e)}"

    return render(request, "dashboard/predict.html", {
        "result": result,
        "actual": actual,
        "confidence": confidence
    })

@login_required
def history_view(request):
    qs = Prediction.objects.filter(user=request.user)

    # Count per class
    class_counts = qs.values('predicted_class').annotate(count=Count('id'))

    labels = [item['predicted_class'] for item in class_counts]
    data = [item['count'] for item in class_counts]

    return render(request, 'dashboard/history.html', {'labels': labels,'data': data,})

@login_required
def profile_page(request):
    profile = request.user.profile
    return render(request, 'dashboard/profile.html', {'profile': profile})

@login_required
def my_predictions(request):
    predictions = Prediction.objects.filter(user=request.user)
    return render(request, 'dashboard/my_predictions.html', {'predictions': predictions})
