import numpy as np
import joblib
import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

dummy = smile.process_signal(
    np.zeros(16000),
    sampling_rate=16000
)

feature_names = list(dummy.columns)

joblib.dump(
    feature_names,
    "feature_names.pkl"
)

print("Saved:", len(feature_names))