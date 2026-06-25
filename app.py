import gradio as gr
import joblib
import opensmile
import numpy as np

# =====================
# LOAD MODELS
# =====================

model = joblib.load("rf_deepfake_final.pkl")
feature_names = list(joblib.load("feature_names.pkl"))

feature_mean = joblib.load("feature_mean.pkl")
feature_std = joblib.load("feature_std.pkl")

# =====================
# FEATURE GROUPING
# =====================

def feature_to_group(name):

    name = name.lower()

    if "f0" in name:
        return "Fundamental frequency (pitch)"

    if "jitter" in name:
        return "Jitter (voice instability)"

    if "shimmer" in name:
        return "Shimmer (amplitude variation)"

    if "loudness" in name:
        return "Loudness"

    if "spectral" in name:
        return "Spectral characteristics"

    if "f1" in name or "f2" in name or "f3" in name:
        return "Formant structure"

    return None


# =====================
# OPENSMILE
# =====================

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# =====================
# PREDICT FUNCTION
# =====================

def predict(audio_path):

    features = smile.process_file(audio_path)

    X = features.values
    x = X[0]

    # ===== prediction =====
    probs = model.predict_proba(X)[0]

    real_prob = float(probs[0]) * 100
    fake_prob = float(probs[1]) * 100

    label = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = max(real_prob, fake_prob)

    # ===== explanation (z-score) =====
    z = np.abs((x - feature_mean) / (feature_std + 1e-8))

    idx = np.argsort(z)[::-1]

    groups = []

    for i in idx:
        if i >= len(feature_names):
            continue

        group = feature_to_group(feature_names[i])

        if group and group not in groups:
            groups.append(group)

        if len(groups) == 5:
            break

    feature_text = "\n\n".join([f"• {g}" for g in groups])

    # ===== output =====
    return f"""
**AUDIO DETECTION by ACOUSTIC FEATURES**

**Prediction:** {label}

**Confidence:** {confidence:.2f}%

**Real probability:** {real_prob:.2f}%

**Fake probability:** {fake_prob:.2f}%

**Main acoustic characteristics of this audio:**

{feature_text}
"""


# =====================
# GRADIO UI
# =====================

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload audio file"),
    outputs=gr.Markdown(),
    flagging_mode="never",
    title="Deepfake Audio Detection",
    description="The system analyzes acoustic features and predicts whether the audio is real or fake."
)

demo.launch(
    server_name="0.0.0.0",
    share=True,
    theme=gr.themes.Soft(primary_hue="violet")
)