import os
import numpy as np
import joblib
import opensmile

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# =========================
# PATHS
# =========================

EXTRACTED_DIR = "/mnt/vas/ext_home/extracted_features"
INWILD_DIR = "/home/ext-khodzhametova@ad.speechpro.com/deepfake_detection"

datasets = {}

# =========================
# LOAD DATASETS
# =========================

datasets["InTheWild"] = {
    "X": joblib.load(os.path.join(INWILD_DIR, "X_features_inwild.pkl")),
    "y": joblib.load(os.path.join(INWILD_DIR, "y_labels_inwild.pkl")),
}

for split in ["train", "dev", "eval"]:
    name = f"ASV2019_{split}"
    datasets[name] = {
        "X": joblib.load(os.path.join(EXTRACTED_DIR, f"{name}_X.pkl")),
        "y": joblib.load(os.path.join(EXTRACTED_DIR, f"{name}_y.pkl")),
    }

for split in ["eval", "hidden", "progress"]:
    name = f"ASV2021_{split}"
    datasets[name] = {
        "X": joblib.load(os.path.join(EXTRACTED_DIR, f"{name}_X.pkl")),
        "y": joblib.load(os.path.join(EXTRACTED_DIR, f"{name}_y.pkl")),
    }

for split in ["train_full", "dev_full"]:
    name = f"ASV5_{split}"
    datasets[name] = {
        "X": joblib.load(os.path.join(EXTRACTED_DIR, f"{name}_X.pkl")),
        "y": joblib.load(os.path.join(EXTRACTED_DIR, f"{name}_y.pkl")),
    }

datasets["ODSS"] = {
    "X": joblib.load(os.path.join(EXTRACTED_DIR, "ODSS_X.pkl")),
    "y": joblib.load(os.path.join(EXTRACTED_DIR, "ODSS_y.pkl")),
}

# =========================
# MERGE DATASETS
# =========================

X_train = np.vstack([d["X"] for d in datasets.values()])
y_train = np.hstack([d["y"] for d in datasets.values()])

print("Train shape:", X_train.shape)

# =========================
# FEATURE STATS (FOR EXPLANATION)
# =========================

feature_mean = np.mean(X_train, axis=0)
feature_std = np.std(X_train, axis=0)

joblib.dump(feature_mean, "feature_mean.pkl")
joblib.dump(feature_std, "feature_std.pkl")

print("Feature statistics saved")

# =========================
# FEATURE NAMES (OPENSMILE FIX)
# =========================

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

dummy = smile.process_signal(
    np.zeros(16000),
    sampling_rate=16000
)

feature_names = list(dummy.columns)

joblib.dump(feature_names, "feature_names.pkl")
print("Feature names saved:", len(feature_names))

# =========================
# CLASS WEIGHTS
# =========================

classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight_dict = dict(zip(classes, class_weights))

# =========================
# MODEL
# =========================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=50,
    min_samples_leaf=25,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1,
)

# =========================
# TRAIN
# =========================

rf.fit(X_train, y_train)

# =========================
# SAVE MODEL
# =========================

joblib.dump(rf, "rf_deepfake_final.pkl")

print("Model saved: rf_deepfake_final.pkl")