import pickle
import numpy as np

# Load and inspect the models
with open('features.pkl', 'w') as log:
    try:
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        log.write(f"Feature names type: {type(feature_names)}\n")
        log.write(f"Feature names: {feature_names}\n")
        if hasattr(feature_names, '__len__'):
            log.write(f"Length: {len(feature_names)}\n")
            log.write(f"Element 0 type: {type(feature_names[0]) if len(feature_names) > 0 else 'empty'}\n")
    except Exception as e:
        log.write(f"Error loading feature_names: {e}\n")

    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        log.write(f"Scaler type: {type(scaler)}\n")
        log.write(f"Scaler mean shape: {scaler.mean_.shape if hasattr(scaler, 'mean_') else 'no mean'}\n")
        log.write(f"Scaler scale shape: {scaler.scale_.shape if hasattr(scaler, 'scale_') else 'no scale'}\n")
    except Exception as e:
        log.write(f"Error loading scaler: {e}\n")

    try:
        with open('policy_predictor.pkl', 'rb') as f:
            model = pickle.load(f)
        log.write(f"Model type: {type(model)}\n")
        if hasattr(model, 'feature_importances_'):
            log.write(f"Feature importances shape: {model.feature_importances_.shape}\n")
        if hasattr(model, 'n_estimators'):
            log.write(f"Number of estimators: {model.n_estimators}\n")
    except Exception as e:
        log.write(f"Error loading model: {e}\n")
