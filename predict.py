import sys
import joblib
import opensmile

# =========================
# MODEL
# =========================

model = joblib.load("rf_deepfake_final.pkl")

# =========================
# OPENSMILE
# =========================

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# =========================
# PREDICTION
# =========================

def predict(audio_path):

    features = smile.process_file(audio_path)
    X = features.values

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    real_prob = float(probs[0]) * 100
    fake_prob = float(probs[1]) * 100

    label = "FAKE" if pred == 1 else "REAL"

    print("\n==============================")
    print("DEEPFAKE DETECTION RESULT")
    print("==============================")
    print(f"Prediction : {label}")
    print(f"Real prob. : {real_prob:.2f}%")
    print(f"Fake prob. : {fake_prob:.2f}%")
    print("==============================\n")

    return label, real_prob, fake_prob


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python3 predict.py path/to/audio.wav")
        sys.exit(1)

    predict(sys.argv[1])