
# import os
# import traceback
# from pathlib import Path
# from uuid import uuid4
# from datetime import datetime

# from flask import Flask, request, jsonify
# from flask_cors import CORS

# import pickle
# import joblib
# import numpy as np
# import pandas as pd
# import re
# import fitz  # PyMuPDF

# # ------------------- Paths ------------------- #
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, "models")
# INPUT_DIR = os.path.join(BASE_DIR, "input")

# os.makedirs(INPUT_DIR, exist_ok=True)

# # ------------------- Flask Setup ------------------- #
# app = Flask(__name__)
# CORS(app)

# # ------------------- Helpers to load models ------------------- #
# def load_pickle(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# def safe_joblib_load(path):
#     try:
#         return joblib.load(path)
#     except Exception as e:
#         app.logger.error(f"joblib load error {path}: {e}")
#         return None

# # ------------------- (Optional) Existing policy model kept as-is ------------------- #
# try:
#     policy_model_path = os.path.join(MODELS_DIR, "policy_predictor.pkl")
#     if os.path.exists(policy_model_path):
#         policy_model = load_pickle(policy_model_path)
#         app.logger.info("✅ Policy model loaded successfully")
#     else:
#         policy_model = None
#         app.logger.info("No policy_predictor found; skipping.")
# except Exception as e:
#     app.logger.error(f"❌ Error loading policy model: {e}")
#     policy_model = None

# try:
#     feature_names_path = os.path.join(MODELS_DIR, "feature_names.pkl")
#     if os.path.exists(feature_names_path):
#         feature_names = load_pickle(feature_names_path)
#         if isinstance(feature_names, (list, tuple, np.ndarray)):
#             feature_names = [str(f) for f in feature_names]
#         else:
#             feature_names = [str(feature_names)]
#         app.logger.info(f"✅ Features loaded: {feature_names}")
#     else:
#         feature_names = []
#         app.logger.info("No feature_names.pkl found; feature list empty.")
# except Exception as e:
#     app.logger.error(f"❌ Error loading feature_names: {e}")
#     feature_names = []

# # ------------------- Contract Analyzer models ------------------- #
# risk_score_model = safe_joblib_load(os.path.join(MODELS_DIR, "risk_score_regressor.pkl"))
# risk_level_model = safe_joblib_load(os.path.join(MODELS_DIR, "risk_level_classifier.pkl"))

# if risk_score_model:
#     app.logger.info("✅ Risk score model loaded.")
# else:
#     app.logger.warning("Risk score model not loaded; contract analysis predictions will be unavailable.")

# if risk_level_model:
#     app.logger.info("✅ Risk level model loaded.")
# else:
#     app.logger.warning("Risk level model not loaded; contract analysis classification will be unavailable.")

# # ------------------- ML extraction code (from your script) ------------------- #
# RISK_KEYWORDS = [
#     "penalty", "breach", "liability", "risk", "exposure", "termination",
#     "confidential", "trade secret", "non-disclosure", "disclosure"
# ]

# def read_pdf_text(file_path: str) -> str:
#     doc = fitz.open(file_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# def extract_features_from_text(text: str) -> dict:
#     # Duration extraction
#     duration_match = re.search(r"Duration[:\-]?\s*(\d+)\s*days", text, re.IGNORECASE)
#     if not duration_match:
#         dates = re.findall(r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
#         if len(dates) >= 2:
#             try:
#                 start = pd.to_datetime(dates[0])
#                 end = pd.to_datetime(dates[1])
#                 duration_days = int((end - start).days)
#             except:
#                 duration_days = 0
#         else:
#             duration_days = 0
#     else:
#         duration_days = int(duration_match.group(1))

#     # Clauses & obligations
#     clauses = re.findall(r"\n\d+\.\s", text)
#     num_clauses = len(clauses)
#     obligations = re.findall(r"\b(shall|must|agree to)\b", text, re.IGNORECASE)
#     num_obligations = len(obligations)

#     # Contract value
#     cv_match = re.search(r"(?:Contract Value|contract value).*?([\d,]+\.\d+)", text, re.IGNORECASE)
#     contract_value = float(cv_match.group(1).replace(",", "")) if cv_match else 0.0

#     # Penalty rate (percentage)
#     pr_match = re.search(r"penalty.*?([\d.,]+)%", text, re.IGNORECASE)
#     penalty_rate = float(pr_match.group(1)) / 100.0 if pr_match else 0.0

#     # Risk keywords count
#     risk_count = sum(1 for kw in RISK_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE))

#     # Derived metrics
#     compliance_score = 80
#     renewal_term_months = 12
#     days_to_expiry = max(duration_days - 30, 0)
#     financial_exposure = contract_value * 0.1 if contract_value > 0 else 0.0
#     breach_probability = min(risk_count * 0.1, 1.0)

#     return {
#         "Duration_Days": duration_days,
#         "Contract_Value": contract_value,
#         "Penalty_Rate": penalty_rate,
#         "Num_Clauses": num_clauses,
#         "Num_Obligations": num_obligations,
#         "Risk_Keywords": risk_count,
#         "Compliance_Score": compliance_score,
#         "Renewal_Term_Months": renewal_term_months,
#         "Days_to_Expiry": days_to_expiry,
#         "Financial_Exposure": financial_exposure,
#         "Breach_Probability": breach_probability,
#         "Region": "LATAM",
#         "Contract_Type": "NDA",
#         "Is_Compliant": 1
#     }

# def features_to_dataframe(features: dict) -> pd.DataFrame:
#     df = pd.DataFrame([features])
#     cols = [
#         "Region", "Contract_Type", "Duration_Days", "Contract_Value", "Penalty_Rate",
#         "Num_Clauses", "Num_Obligations", "Risk_Keywords", "Compliance_Score",
#         "Renewal_Term_Months", "Days_to_Expiry", "Financial_Exposure", "Breach_Probability", "Is_Compliant"
#     ]
#     for c in cols:
#         if c not in df.columns:
#             df[c] = 0
#     return df[cols]

# def generate_summary(features: dict, risk_score: float, risk_level: str) -> str:
#     summary = f"""
# This contract spans {features['Duration_Days']} days with {features['Num_Clauses']} clauses and {features['Num_Obligations']} obligations.
# The total contract value is estimated at {features['Contract_Value']:.2f} AED with a penalty rate of {features['Penalty_Rate']:.2%}.
# Financial exposure is calculated at {features['Financial_Exposure']:.2f} AED.
# Risk analysis detected {features['Risk_Keywords']} key risk factors.
# Overall Risk Score predicted by the model is {risk_score:.2f}, corresponding to a risk level of '{risk_level}'.
# Compliance score is approximately {features['Compliance_Score']}.
# This summary is intended to give a quick understanding of the contract's risk and obligations.
# """
#     return " ".join(line.strip() for line in summary.strip().split("\n"))

# # ------------------- Routes ------------------- #
# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "healthy"})

# @app.route("/features", methods=["GET"])
# def get_features():
#     return jsonify({"features": feature_names})

# @app.route("/predict", methods=["POST"])
# def predict():
#     # keep original predict endpoint (policy model) for backward compatibility
#     if policy_model is None:
#         return jsonify({"error": "Policy model not loaded"}), 500
#     try:
#         data = request.get_json(force=True)
#         feature_values = []
#         missing = []
#         for f in feature_names:
#             if f not in data:
#                 missing.append(f)
#                 continue
#             try:
#                 feature_values.append(float(data[f]))
#             except Exception:
#                 return jsonify({"error": f"Invalid value for {f}"}), 400
#         if missing:
#             return jsonify({"error": "Missing features", "missing": missing}), 400
#         arr = np.array(feature_values, dtype=float).reshape(1, -1)
#         pred = policy_model.predict(arr)
#         return jsonify({"prediction": float(pred[0]), "status": "success"})
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# @app.route("/analyze-contract", methods=["POST"])
# def analyze_contract():
#     """
#     Accepts multipart/form-data with 'file' (PDF).
#     Returns JSON: risk_score, risk_level, summary, features, file_saved_as
#     """
#     try:
#         if "file" not in request.files:
#             return jsonify({"error": "No file part in request"}), 400

#         file = request.files["file"]
#         if file.filename == "":
#             return jsonify({"error": "No selected file"}), 400

#         if not file.filename.lower().endswith(".pdf"):
#             return jsonify({"error": "Only PDF files are supported"}), 400

#         # Save uploaded file
#         unique_id = uuid4().hex
#         timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
#         safe_name = f"{timestamp}_{unique_id}_{file.filename}"
#         input_path = os.path.join(INPUT_DIR, safe_name)
#         file.save(input_path)
#         app.logger.info(f"Saved uploaded file to {input_path}")

#         # Extract text & features
#         text = read_pdf_text(input_path)
#         if not text.strip():
#             return jsonify({"error": "No text extracted from PDF. If scanned, run OCR first."}), 400

#         features = extract_features_from_text(text)
#         df_features = features_to_dataframe(features)

#         # Predictions
#         risk_score = None
#         risk_level = None
#         if risk_score_model is not None:
#             try:
#                 risk_score = float(risk_score_model.predict(df_features)[0])
#             except Exception as e:
#                 app.logger.error(f"Risk score prediction error: {e}")
#         if risk_level_model is not None:
#             try:
#                 risk_level = str(risk_level_model.predict(df_features)[0])
#             except Exception as e:
#                 app.logger.error(f"Risk level prediction error: {e}")

#         # NLP summary
#         summary = generate_summary(features, risk_score if risk_score is not None else 0.0,
#                                    risk_level if risk_level is not None else "Unknown")

#         response = {
#             "status": "success",
#             "risk_score": risk_score,
#             "risk_level": risk_level,
#             "summary": summary,
#             "features": features,
#             "file_saved_as": safe_name
#         }
#         return jsonify(response)

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# # ------------------- Run ------------------- #
# if __name__ == "__main__":
#     # debug only for development
#     app.run(host="0.0.0.0", port=5000, debug=True)
# app.py
# import os
# import traceback
# from pathlib import Path
# from uuid import uuid4
# from datetime import datetime

# from flask import Flask, request, jsonify
# from flask_cors import CORS

# import pickle
# import joblib
# import numpy as np
# import pandas as pd
# import re
# import fitz  # PyMuPDF
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, r2_score

# # ------------------- Paths & Directories ------------------- #
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, "models")
# INPUT_DIR = os.path.join(BASE_DIR, "input")
# DATA_PATH = os.path.join(BASE_DIR, "backend", "supply_chain_dataset_learnable.csv")  # adjust if needed

# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(INPUT_DIR, exist_ok=True)

# # ------------------- Flask Setup ------------------- #
# app = Flask(__name__)
# CORS(app)

# # ------------------- Helpers to load models ------------------- #
# def load_pickle(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# def safe_joblib_load(path):
#     try:
#         return joblib.load(path)
#     except Exception as e:
#         app.logger.error(f"joblib load error {path}: {e}")
#         return None

# # ------------------- (Optional) Existing policy & contract models ------------------- #
# # Policy model (kept for backward compatibility in your code)
# try:
#     policy_model_path = os.path.join(MODELS_DIR, "policy_predictor.pkl")
#     if os.path.exists(policy_model_path):
#         policy_model = load_pickle(policy_model_path)
#         app.logger.info("✅ Policy model loaded successfully")
#     else:
#         policy_model = None
#         app.logger.info("No policy_predictor found; skipping.")
# except Exception as e:
#     app.logger.error(f"❌ Error loading policy model: {e}")
#     policy_model = None

# # Feature names for policy endpoint (if present)
# try:
#     feature_names_path = os.path.join(MODELS_DIR, "feature_names.pkl")
#     if os.path.exists(feature_names_path):
#         feature_names = load_pickle(feature_names_path)
#         if isinstance(feature_names, (list, tuple, np.ndarray)):
#             feature_names = [str(f) for f in feature_names]
#         else:
#             feature_names = [str(feature_names)]
#         app.logger.info(f"Features loaded: {feature_names}")
#     else:
#         feature_names = []
#         app.logger.info("No feature_names.pkl found; feature list empty.")
# except Exception as e:
#     app.logger.error(f"Error loading feature_names: {e}")
#     feature_names = []

# # Contract Analyzer models (from your provided code)
# risk_score_model = safe_joblib_load(os.path.join(MODELS_DIR, "risk_score_regressor.pkl"))
# risk_level_model = safe_joblib_load(os.path.join(MODELS_DIR, "risk_level_classifier.pkl"))

# if risk_score_model:
#     app.logger.info("Risk score model loaded.")
# else:
#     app.logger.warning("Risk score model not loaded; contract analysis predictions will be unavailable.")

# if risk_level_model:
#     app.logger.info("Risk level model loaded.")
# else:
#     app.logger.warning("Risk level model not loaded; contract analysis classification will be unavailable.")

# # ------------------- Contract extraction helpers (kept) ------------------- #
# RISK_KEYWORDS = [
#     "penalty", "breach", "liability", "risk", "exposure", "termination",
#     "confidential", "trade secret", "non-disclosure", "disclosure"
# ]

# def read_pdf_text(file_path: str) -> str:
#     doc = fitz.open(file_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# def extract_features_from_text(text: str) -> dict:
#     # Duration extraction
#     duration_match = re.search(r"Duration[:\-]?\s*(\d+)\s*days", text, re.IGNORECASE)
#     if not duration_match:
#         dates = re.findall(r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
#         if len(dates) >= 2:
#             try:
#                 start = pd.to_datetime(dates[0])
#                 end = pd.to_datetime(dates[1])
#                 duration_days = int((end - start).days)
#             except:
#                 duration_days = 0
#         else:
#             duration_days = 0
#     else:
#         duration_days = int(duration_match.group(1))

#     # Clauses & obligations
#     clauses = re.findall(r"\n\d+\.\s", text)
#     num_clauses = len(clauses)
#     obligations = re.findall(r"\b(shall|must|agree to)\b", text, re.IGNORECASE)
#     num_obligations = len(obligations)

#     # Contract value
#     cv_match = re.search(r"(?:Contract Value|contract value).*?([\d,]+\.\d+)", text, re.IGNORECASE)
#     contract_value = float(cv_match.group(1).replace(",", "")) if cv_match else 0.0

#     # Penalty rate (percentage)
#     pr_match = re.search(r"penalty.*?([\d.,]+)%", text, re.IGNORECASE)
#     penalty_rate = float(pr_match.group(1).replace(",", "")) / 100.0 if pr_match else 0.0

#     # Risk keywords count
#     risk_count = sum(1 for kw in RISK_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE))

#     # Derived metrics (simple heuristics; you can replace with improved logic)
#     compliance_score = 80
#     renewal_term_months = 12
#     days_to_expiry = max(duration_days - 30, 0)
#     financial_exposure = contract_value * 0.1 if contract_value > 0 else 0.0
#     breach_probability = min(risk_count * 0.1, 1.0)

#     return {
#         "Duration_Days": duration_days,
#         "Contract_Value": contract_value,
#         "Penalty_Rate": penalty_rate,
#         "Num_Clauses": num_clauses,
#         "Num_Obligations": num_obligations,
#         "Risk_Keywords": risk_count,
#         "Compliance_Score": compliance_score,
#         "Renewal_Term_Months": renewal_term_months,
#         "Days_to_Expiry": days_to_expiry,
#         "Financial_Exposure": financial_exposure,
#         "Breach_Probability": breach_probability,
#         "Region": "LATAM",
#         "Contract_Type": "NDA",
#         "Is_Compliant": 1
#     }

# def features_to_dataframe(features: dict) -> pd.DataFrame:
#     df = pd.DataFrame([features])
#     cols = [
#         "Region", "Contract_Type", "Duration_Days", "Contract_Value", "Penalty_Rate",
#         "Num_Clauses", "Num_Obligations", "Risk_Keywords", "Compliance_Score",
#         "Renewal_Term_Months", "Days_to_Expiry", "Financial_Exposure", "Breach_Probability", "Is_Compliant"
#     ]
#     for c in cols:
#         if c not in df.columns:
#             df[c] = 0
#     return df[cols]

# def generate_summary(features: dict, risk_score: float, risk_level: str) -> str:
#     summary = f"""
# This contract spans {features['Duration_Days']} days with {features['Num_Clauses']} clauses and {features['Num_Obligations']} obligations.
# The total contract value is estimated at {features['Contract_Value']:.2f} AED with a penalty rate of {features['Penalty_Rate']:.2%}.
# Financial exposure is calculated at {features['Financial_Exposure']:.2f} AED.
# Risk analysis detected {features['Risk_Keywords']} key risk factors.
# Overall Risk Score predicted by the model is {risk_score:.2f}, corresponding to a risk level of '{risk_level}'.
# Compliance score is approximately {features['Compliance_Score']}.
# This summary is intended to give a quick understanding of the contract's risk and obligations.
# """
#     return " ".join(line.strip() for line in summary.strip().split("\n"))

# # ------------------- Supply Chain Model Implementation ------------------- #
# def calculate_stockout_probability(stock_level, inventory_threshold):
#     try:
#         if inventory_threshold <= 0:
#             return 0.0
#         return round(max(0.0, (inventory_threshold - stock_level) / float(inventory_threshold)), 2)
#     except Exception:
#         return 0.0

# def calculate_reorder_quantity(stock_level, inventory_threshold, safety_stock=20):
#     try:
#         return int(max(0, inventory_threshold - stock_level + safety_stock))
#     except Exception:
#         return 0

# def calculate_total_cost(order_quantity, cost_per_unit, delay_probability=0.0, delay_penalty_per_day=10):
#     try:
#         expected_delay_cost = delay_probability * delay_penalty_per_day
#         return order_quantity * cost_per_unit + expected_delay_cost
#     except Exception:
#         return order_quantity * cost_per_unit

# def generate_summary_supply(pred_df):
#     summaries = []
#     for _, row in pred_df.iterrows():
#         delay_risk = row.get('Delay_Probability', 0.0) * 100
#         delay_days = round(row.get('Expected_Delay_Days', 0.0), 1)
#         stockout = row.get('Stockout_Probability', 0.0)
#         reorder_qty = int(row.get('Recommended_Reorder_Qty', 0))
#         total_cost = row.get('Total_Cost_Impact', 0.0)
#         supplier_reliability = row.get('Supplier_Reliability_Score', 0.0)
#         order_id = int(row.get('Order_ID', -1))

#         # Delay interpretation
#         if delay_risk < 10:
#             delay_msg = "This order is on track for timely delivery."
#         elif delay_risk < 50:
#             delay_msg = "There’s a moderate risk of delay — monitor supplier performance closely."
#         else:
#             delay_msg = "High delay risk detected — consider alternate suppliers or expediting the order."

#         # Stockout interpretation
#         if stockout == 0:
#             stock_msg = "Current inventory levels are sufficient to meet demand."
#         elif stockout < 0.4:
#             stock_msg = "Slight stock pressure detected — review reorder plans soon."
#         else:
#             stock_msg = "Critical stock shortage risk — reorder immediately to prevent supply disruptions."

#         # Cost insight
#         if total_cost < 50000:
#             cost_msg = "Cost impact is minimal, within acceptable range."
#         elif total_cost < 100000:
#             cost_msg = "Cost impact is moderate; optimize supplier contracts to save costs."
#         else:
#             cost_msg = "High total cost impact — investigate procurement and logistics expenses."

#         summary = (
#             f"Order {order_id} Analysis:\n"
#             f"- 📦 Delay Probability: {delay_risk:.1f}% ({delay_msg})\n"
#             f"- ⏱ Expected Delay Duration: {delay_days} day(s)\n"
#             f"- 🧮 Supplier Reliability: {supplier_reliability:.2f}\n"
#             f"- 📉 Stockout Probability: {stockout}\n"
#             f"- 🔁 Recommended Reorder Quantity: {reorder_qty}\n"
#             f"- 💰 Estimated Total Cost Impact: {total_cost:,.2f}\n"
#             f"💡 {stock_msg}\n"
#             f"💼 {cost_msg}"
#         )
#         summaries.append(summary)
#     return summaries

# class SupplyChainModel:
#     def __init__(self):
#         self.delay_model = None
#         self.delay_days_model = None
#         self.encoder_supplier = LabelEncoder()
#         self.encoder_product = LabelEncoder()
#         # attempt to load models if present
#         try:
#             self.delay_model = safe_joblib_load(os.path.join(MODELS_DIR, "delay_model.pkl"))
#             self.delay_days_model = safe_joblib_load(os.path.join(MODELS_DIR, "delay_days_model.pkl"))
#             self.encoder_supplier = safe_joblib_load(os.path.join(MODELS_DIR, "supplier_encoder.pkl")) or LabelEncoder()
#             self.encoder_product = safe_joblib_load(os.path.join(MODELS_DIR, "product_encoder.pkl")) or LabelEncoder()
#             app.logger.info("Supply chain models and encoders loaded (if available).")
#         except Exception as e:
#             app.logger.warning(f"Could not pre-load supply chain models: {e}")

#     def preprocess(self, df, fit_encoders=True):
#         df = df.copy()
#         # ensure required columns exist
#         for c in ['Supplier', 'Product_SKU']:
#             if c not in df.columns:
#                 df[c] = "UNKNOWN"

#         if fit_encoders:
#             # fit LabelEncoders on training data
#             self.encoder_supplier.fit(df['Supplier'].astype(str).fillna("UNKNOWN"))
#             self.encoder_product.fit(df['Product_SKU'].astype(str).fillna("UNKNOWN"))
#         # transform with careful handling of unseen labels
#         def safe_transform(encoder, series):
#             arr = []
#             classes = list(encoder.classes_) if hasattr(encoder, "classes_") else []
#             for v in series.astype(str).fillna("UNKNOWN"):
#                 if v in classes:
#                     arr.append(int(np.where(np.array(classes) == v)[0][0]))
#                 else:
#                     arr.append(-1)
#             return arr

#         df['Supplier_enc'] = safe_transform(self.encoder_supplier, df['Supplier'])
#         df['Product_SKU_enc'] = safe_transform(self.encoder_product, df['Product_SKU'])

#         # fill numeric columns if missing
#         numeric_defaults = {
#             'Order_Quantity': 0, 'Lead_Time': 0, 'Stock_Level': 0,
#             'Inventory_Threshold': 0, 'Cost': 0.0, 'Supplier_Reliability_Score': 0.0
#         }
#         for k, v in numeric_defaults.items():
#             if k not in df.columns:
#                 df[k] = v
#             df[k] = pd.to_numeric(df[k], errors='coerce').fillna(v)

#         return df

#     def train(self, data_path=DATA_PATH):
#         if not os.path.exists(data_path):
#             raise FileNotFoundError(f"Training CSV not found at {data_path}")

#         df = pd.read_csv(data_path)
#         df = df.copy()
#         # ensure columns expected exist
#         required = ['Order_Quantity', 'Lead_Time', 'Stock_Level', 'Inventory_Threshold',
#                     'Cost', 'Supplier_Reliability_Score', 'Supplier', 'Product_SKU', 'Delay_Risk', 'Delay_Days']
#         for c in required:
#             if c not in df.columns:
#                 raise ValueError(f"Required column missing in training data: {c}")

#         df = self.preprocess(df, fit_encoders=True)

#         features = ['Order_Quantity', 'Lead_Time', 'Stock_Level', 'Inventory_Threshold',
#                     'Cost', 'Supplier_Reliability_Score', 'Supplier_enc', 'Product_SKU_enc']
#         X = df[features]
#         y_delay = df['Delay_Risk']
#         y_days = df['Delay_Days']

#         X_train, X_test, y_train_delay, y_test_delay, y_train_days, y_test_days = train_test_split(
#             X, y_delay, y_days, test_size=0.2, random_state=42
#         )

#         # Tuned models for high accuracy
#         self.delay_model = RandomForestClassifier(
#             n_estimators=400,
#             max_depth=15,
#             min_samples_split=4,
#             random_state=42,
#             class_weight='balanced'
#         )
#         self.delay_model.fit(X_train, y_train_delay)

#         self.delay_days_model = GradientBoostingRegressor(
#             n_estimators=400,
#             max_depth=6,
#             learning_rate=0.05,
#             random_state=42
#         )
#         self.delay_days_model.fit(X_train, y_train_days)

#         # Quick validation
#         y_pred_class = self.delay_model.predict(X_test)
#         y_pred_days = self.delay_days_model.predict(X_test)
#         acc = accuracy_score(y_test_delay, y_pred_class)
#         r2 = r2_score(y_test_days, y_pred_days)

#         # Save models and encoders
#         joblib.dump(self.delay_model, os.path.join(MODELS_DIR, "delay_model.pkl"))
#         joblib.dump(self.delay_days_model, os.path.join(MODELS_DIR, "delay_days_model.pkl"))
#         joblib.dump(self.encoder_supplier, os.path.join(MODELS_DIR, "supplier_encoder.pkl"))
#         joblib.dump(self.encoder_product, os.path.join(MODELS_DIR, "product_encoder.pkl"))

#         return {"accuracy": float(acc), "r2": float(r2)}

#     def load_models(self):
#         self.delay_model = safe_joblib_load(os.path.join(MODELS_DIR, "delay_model.pkl"))
#         self.delay_days_model = safe_joblib_load(os.path.join(MODELS_DIR, "delay_days_model.pkl"))
#         loaded_supplier = safe_joblib_load(os.path.join(MODELS_DIR, "supplier_encoder.pkl"))
#         loaded_product = safe_joblib_load(os.path.join(MODELS_DIR, "product_encoder.pkl"))
#         if loaded_supplier is not None:
#             self.encoder_supplier = loaded_supplier
#         if loaded_product is not None:
#             self.encoder_product = loaded_product

#     def predict(self, df):
#         df = df.copy()
#         df = self.preprocess(df, fit_encoders=False)

#         features = ['Order_Quantity','Lead_Time','Stock_Level','Inventory_Threshold',
#                     'Cost','Supplier_Reliability_Score','Supplier_enc','Product_SKU_enc']
#         X = df[features].astype(float).fillna(0.0)

#         if self.delay_model is None or self.delay_days_model is None:
#             raise RuntimeError("Supply chain models are not trained/loaded.")

#         # predict probability for delay (if classifier supports predict_proba)
#         try:
#             df['Delay_Probability'] = self.delay_model.predict_proba(X)[:, 1]
#         except Exception:
#             # fallback to predict (0/1)
#             df['Delay_Probability'] = self.delay_model.predict(X).astype(float)

#         df['Expected_Delay_Days'] = np.maximum(0, self.delay_days_model.predict(X))

#         # Utilities
#         df['Stockout_Probability'] = df.apply(
#             lambda r: calculate_stockout_probability(r['Stock_Level'], r['Inventory_Threshold']), axis=1)
#         df['Recommended_Reorder_Qty'] = df.apply(
#             lambda r: calculate_reorder_quantity(r['Stock_Level'], r['Inventory_Threshold']), axis=1)
#         df['Total_Cost_Impact'] = df.apply(
#             lambda r: calculate_total_cost(r['Order_Quantity'], r['Cost'], r['Delay_Probability']), axis=1)

#         # NLP Summaries
#         df['NLP_Summary'] = generate_summary_supply(df)
#         # ensure Order_ID present
#         if 'Order_ID' not in df.columns:
#             df['Order_ID'] = -1

#         # convert numpy types to native for JSON
#         return df

# # instantiate a singleton SCM
# scm = SupplyChainModel()

# # ------------------- Routes ------------------- #
# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "healthy"})

# @app.route("/features", methods=["GET"])
# def get_features():
#     return jsonify({"features": feature_names})

# @app.route("/predict", methods=["POST"])
# def predict():
#     # keep original predict endpoint (policy model) for backward compatibility
#     if policy_model is None:
#         return jsonify({"error": "Policy model not loaded"}), 500
#     try:
#         data = request.get_json(force=True)
#         feature_values = []
#         missing = []
#         for f in feature_names:
#             if f not in data:
#                 missing.append(f)
#                 continue
#             try:
#                 feature_values.append(float(data[f]))
#             except Exception:
#                 return jsonify({"error": f"Invalid value for {f}"}), 400
#         if missing:
#             return jsonify({"error": "Missing features", "missing": missing}), 400
#         arr = np.array(feature_values, dtype=float).reshape(1, -1)
#         pred = policy_model.predict(arr)
#         return jsonify({"prediction": float(pred[0]), "status": "success"})
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# @app.route("/analyze-contract", methods=["POST"])
# def analyze_contract():
#     """
#     Accepts multipart/form-data with 'file' (PDF).
#     Returns JSON: risk_score, risk_level, summary, features, file_saved_as
#     """
#     try:
#         if "file" not in request.files:
#             return jsonify({"error": "No file part in request"}), 400

#         file = request.files["file"]
#         if file.filename == "":
#             return jsonify({"error": "No selected file"}), 400

#         if not file.filename.lower().endswith(".pdf"):
#             return jsonify({"error": "Only PDF files are supported"}), 400

#         # Save uploaded file
#         unique_id = uuid4().hex
#         timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
#         safe_name = f"{timestamp}_{unique_id}_{file.filename}"
#         input_path = os.path.join(INPUT_DIR, safe_name)
#         file.save(input_path)
#         app.logger.info(f"Saved uploaded file to {input_path}")

#         # Extract text & features
#         text = read_pdf_text(input_path)
#         if not text.strip():
#             return jsonify({"error": "No text extracted from PDF. If scanned, run OCR first."}), 400

#         features = extract_features_from_text(text)
#         df_features = features_to_dataframe(features)

#         # Predictions
#         risk_score = None
#         risk_level = None
#         if risk_score_model is not None:
#             try:
#                 risk_score = float(risk_score_model.predict(df_features)[0])
#             except Exception as e:
#                 app.logger.error(f"Risk score prediction error: {e}")
#         if risk_level_model is not None:
#             try:
#                 risk_level = str(risk_level_model.predict(df_features)[0])
#             except Exception as e:
#                 app.logger.error(f"Risk level prediction error: {e}")

#         # NLP summary
#         summary = generate_summary(features, risk_score if risk_score is not None else 0.0,
#                                    risk_level if risk_level is not None else "Unknown")

#         response = {
#             "status": "success",
#             "risk_score": risk_score,
#             "risk_level": risk_level,
#             "summary": summary,
#             "features": features,
#             "file_saved_as": safe_name
#         }
#         return jsonify(response)

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# # ---------------- Supply Chain Endpoints ---------------- #
# @app.route("/supplychain/train", methods=["POST"])
# def supplychain_train():
#     """
#     Triggers training of the supply chain models using the CSV at DATA_PATH.
#     Returns training metrics.
#     """
#     try:
#         metrics = scm.train(data_path=DATA_PATH)
#         return jsonify({"status": "success", "metrics": metrics})
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"status": "error", "detail": str(e)}), 500

# @app.route("/supplychain/predict-file", methods=["POST"])
# def supplychain_predict_file():
#     """
#     Accepts multipart/form-data with 'file' (CSV).
#     The CSV should contain columns like:
#     Order_ID,Supplier,Product_SKU,Order_Quantity,Lead_Time,Stock_Level,Inventory_Threshold,Cost,Supplier_Reliability_Score
#     """
#     try:
#         if "file" not in request.files:
#             return jsonify({"error": "No file part in request"}), 400

#         file = request.files["file"]
#         if file.filename == "":
#             return jsonify({"error": "No selected file"}), 400

#         if not (file.filename.lower().endswith(".csv")):
#             return jsonify({"error": "Only CSV files are supported"}), 400

#         # Save file
#         unique_id = uuid4().hex
#         timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
#         safe_name = f"{timestamp}_{unique_id}_{file.filename}"
#         input_path = os.path.join(INPUT_DIR, safe_name)
#         file.save(input_path)
#         app.logger.info(f"Saved uploaded CSV to {input_path}")

#         # read CSV
#         df = pd.read_csv(input_path)

#         # Ensure models loaded
#         scm.load_models()
#         if scm.delay_model is None or scm.delay_days_model is None:
#             return jsonify({"error": "Supply chain models are not trained. Call /supplychain/train first."}), 400

#         preds = scm.predict(df)

#         # prepare JSON-friendly output
#         preds_out = preds.to_dict(orient="records")
#         for r in preds_out:
#             # turn numpy types to python natives
#             for k, v in r.copy().items():
#                 if isinstance(v, (np.integer, np.floating)):
#                     r[k] = v.item()
#                 elif isinstance(v, np.ndarray):
#                     r[k] = v.tolist()

#         return jsonify({"status": "success", "predictions": preds_out})
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"status": "error", "detail": str(e)}), 500

# @app.route("/supplychain/predict-json", methods=["POST"])
# def supplychain_predict_json():
#     """
#     Accepts application/json body containing either a single object or a list of order objects.
#     Returns predictions for each order.
#     """
#     try:
#         data = request.get_json(force=True)
#         # wrap single object in list
#         if isinstance(data, dict):
#             orders = [data]
#         elif isinstance(data, list):
#             orders = data
#         else:
#             return jsonify({"error": "Invalid payload"}), 400

#         df = pd.DataFrame(orders)

#         scm.load_models()
#         if scm.delay_model is None or scm.delay_days_model is None:
#             return jsonify({"error": "Supply chain models are not trained. Call /supplychain/train first."}), 400

#         preds = scm.predict(df)
#         preds_out = preds.to_dict(orient="records")
#         for r in preds_out:
#             for k, v in r.copy().items():
#                 if isinstance(v, (np.integer, np.floating)):
#                     r[k] = v.item()
#                 elif isinstance(v, np.ndarray):
#                     r[k] = v.tolist()

#         return jsonify({"status": "success", "predictions": preds_out})
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"status": "error", "detail": str(e)}), 500
    

# # ---------------- Supply Chain Models & Singleton ---------------- #
# scm = SupplyChainModel()

# # ---------------- Supply Chain Endpoints ---------------- #
# @app.route("/supplychain/train", methods=["POST"])
# def supplychain_train():
#     """
#     Triggers training of the supply chain models using the CSV at DATA_PATH.
#     Returns training metrics.
#     """
#     try:
#         metrics = scm.train(data_path=DATA_PATH)
#         return jsonify({"status": "success", "metrics": metrics})
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"status": "error", "detail": str(e)}), 500

# @app.route("/supplychain/predict-file", methods=["POST"])
# def supplychain_predict_file():
#     """
#     Accepts multipart/form-data with 'file' (CSV).
#     The CSV should contain columns like:
#     Order_ID,Supplier,Product_SKU,Order_Quantity,Lead_Time,Stock_Level,Inventory_Threshold,Cost,Supplier_Reliability_Score
#     """
#     try:
#         if "file" not in request.files:
#             return jsonify({"error": "No file part in request"}), 400

#         file = request.files["file"]
#         if file.filename == "":
#             return jsonify({"error": "No selected file"}), 400

#         if not (file.filename.lower().endswith(".csv")):
#             return jsonify({"error": "Only CSV files are supported"}), 400

#         # Save file
#         unique_id = uuid4().hex
#         timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
#         safe_name = f"{timestamp}_{unique_id}_{file.filename}"
#         input_path = os.path.join(INPUT_DIR, safe_name)
#         file.save(input_path)
#         app.logger.info(f"Saved uploaded CSV to {input_path}")

#         # read CSV
#         df = pd.read_csv(input_path)

#         # Ensure models loaded
#         scm.load_models()
#         if scm.delay_model is None or scm.delay_days_model is None:
#             return jsonify({"error": "Supply chain models are not trained. Call /supplychain/train first."}), 400

#         preds = scm.predict(df)

#         # prepare JSON-friendly output
#         preds_out = preds.to_dict(orient="records")
#         for r in preds_out:
#             # turn numpy types to python natives
#             for k, v in r.copy().items():
#                 if isinstance(v, (np.integer, np.floating)):
#                     r[k] = v.item()
#                 elif isinstance(v, np.ndarray):
#                     r[k] = v.tolist()

#         return jsonify({"status": "success", "predictions": preds_out})
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"status": "error", "detail": str(e)}), 500

# @app.route("/supplychain/predict-json", methods=["POST"])
# def supplychain_predict_json():
#     """
#     Accepts application/json body containing either a single object or a list of order objects.
#     Returns predictions for each order.
#     """
#     try:
#         data = request.get_json(force=True)
#         # wrap single object in list
#         if isinstance(data, dict):
#             orders = [data]
#         elif isinstance(data, list):
#             orders = data
#         else:
#             return jsonify({"error": "Invalid payload"}), 400

#         df = pd.DataFrame(orders)

#         scm.load_models()
#         if scm.delay_model is None or scm.delay_days_model is None:
#             return jsonify({"error": "Supply chain models are not trained. Call /supplychain/train first."}), 400

#         preds = scm.predict(df)
#         preds_out = preds.to_dict(orient="records")
#         for r in preds_out:
#             for k, v in r.copy().items():
#                 if isinstance(v, (np.integer, np.floating)):
#                     r[k] = v.item()
#                 elif isinstance(v, np.ndarray):
#                     r[k] = v.tolist()

#         return jsonify({"status": "success", "predictions": preds_out})
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"status": "error", "detail": str(e)}), 500


# # ------------------- Run ------------------- #
# if __name__ == "__main__":
#     # debug only for development - do not use debug=True in production
#     app.run(host="0.0.0.0", port=5000, debug=True)
# # app.py
# import os
# import traceback
# from pathlib import Path
# from uuid import uuid4
# from datetime import datetime

# from flask import Flask, request, jsonify
# from flask_cors import CORS

# import pickle
# import joblib
# import numpy as np
# import pandas as pd
# import re
# import fitz  # PyMuPDF
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, r2_score

# # ------------------- Paths & Directories ------------------- #
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, "models")
# INPUT_DIR = os.path.join(BASE_DIR, "input")
# DATA_PATH = os.path.join(BASE_DIR, "backend", "supply_chain_dataset_learnable.csv")  # adjust if needed

# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(INPUT_DIR, exist_ok=True)

# # ------------------- Flask Setup ------------------- #
# app = Flask(__name__)
# CORS(app)

# # ------------------- Helpers ------------------- #
# def load_pickle(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# def safe_joblib_load(path):
#     try:
#         return joblib.load(path)
#     except Exception as e:
#         app.logger.error(f"joblib load error {path}: {e}")
#         return None

# # ------------------- Policy & Contract Models ------------------- #
# try:
#     policy_model_path = os.path.join(MODELS_DIR, "policy_predictor.pkl")
#     policy_model = load_pickle(policy_model_path) if os.path.exists(policy_model_path) else None
# except Exception as e:
#     policy_model = None

# try:
#     feature_names_path = os.path.join(MODELS_DIR, "feature_names.pkl")
#     feature_names = load_pickle(feature_names_path) if os.path.exists(feature_names_path) else []
# except Exception as e:
#     feature_names = []

# # Contract Analyzer models
# risk_score_model = safe_joblib_load(os.path.join(MODELS_DIR, "risk_score_regressor.pkl"))
# risk_level_model = safe_joblib_load(os.path.join(MODELS_DIR, "risk_level_classifier.pkl"))

# # ------------------- Contract Extraction Helpers ------------------- #
# RISK_KEYWORDS = [
#     "penalty", "breach", "liability", "risk", "exposure", "termination",
#     "confidential", "trade secret", "non-disclosure", "disclosure"
# ]

# def read_pdf_text(file_path: str) -> str:
#     doc = fitz.open(file_path)
#     return "".join(page.get_text() for page in doc)

# def extract_features_from_text(text: str) -> dict:
#     duration_match = re.search(r"Duration[:\-]?\s*(\d+)\s*days", text, re.IGNORECASE)
#     if not duration_match:
#         dates = re.findall(r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
#         if len(dates) >= 2:
#             try:
#                 start = pd.to_datetime(dates[0])
#                 end = pd.to_datetime(dates[1])
#                 duration_days = int((end - start).days)
#             except:
#                 duration_days = 0
#         else:
#             duration_days = 0
#     else:
#         duration_days = int(duration_match.group(1))

#     clauses = re.findall(r"\n\d+\.\s", text)
#     obligations = re.findall(r"\b(shall|must|agree to)\b", text, re.IGNORECASE)
#     cv_match = re.search(r"(?:Contract Value|contract value).*?([\d,]+\.\d+)", text, re.IGNORECASE)
#     pr_match = re.search(r"penalty.*?([\d.,]+)%", text, re.IGNORECASE)
#     risk_count = sum(1 for kw in RISK_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE))

#     contract_value = float(cv_match.group(1).replace(",", "")) if cv_match else 0.0
#     penalty_rate = float(pr_match.group(1).replace(",", "")) / 100.0 if pr_match else 0.0

#     compliance_score = 80
#     renewal_term_months = 12
#     days_to_expiry = max(duration_days - 30, 0)
#     financial_exposure = contract_value * 0.1 if contract_value > 0 else 0.0
#     breach_probability = min(risk_count * 0.1, 1.0)

#     return {
#         "Duration_Days": duration_days,
#         "Contract_Value": contract_value,
#         "Penalty_Rate": penalty_rate,
#         "Num_Clauses": len(clauses),
#         "Num_Obligations": len(obligations),
#         "Risk_Keywords": risk_count,
#         "Compliance_Score": compliance_score,
#         "Renewal_Term_Months": renewal_term_months,
#         "Days_to_Expiry": days_to_expiry,
#         "Financial_Exposure": financial_exposure,
#         "Breach_Probability": breach_probability,
#         "Region": "LATAM",
#         "Contract_Type": "NDA",
#         "Is_Compliant": 1
#     }

# def features_to_dataframe(features: dict) -> pd.DataFrame:
#     df = pd.DataFrame([features])
#     cols = [
#         "Region", "Contract_Type", "Duration_Days", "Contract_Value", "Penalty_Rate",
#         "Num_Clauses", "Num_Obligations", "Risk_Keywords", "Compliance_Score",
#         "Renewal_Term_Months", "Days_to_Expiry", "Financial_Exposure", "Breach_Probability", "Is_Compliant"
#     ]
#     for c in cols:
#         if c not in df.columns:
#             df[c] = 0
#     return df[cols]

# def generate_summary(features: dict, risk_score: float, risk_level: str) -> str:
#     summary = f"""
# This contract spans {features['Duration_Days']} days with {features['Num_Clauses']} clauses and {features['Num_Obligations']} obligations.
# The total contract value is estimated at {features['Contract_Value']:.2f} AED with a penalty rate of {features['Penalty_Rate']:.2%}.
# Financial exposure is calculated at {features['Financial_Exposure']:.2f} AED.
# Risk analysis detected {features['Risk_Keywords']} key risk factors.
# Overall Risk Score predicted by the model is {risk_score:.2f}, corresponding to a risk level of '{risk_level}'.
# Compliance score is approximately {features['Compliance_Score']}.
# This summary is intended to give a quick understanding of the contract's risk and obligations.
# """
#     return " ".join(line.strip() for line in summary.strip().split("\n"))

# # ------------------- Supply Chain Model ------------------- #
# def calculate_stockout_probability(stock_level, inventory_threshold):
#     return round(max(0.0, (inventory_threshold - stock_level) / max(1, inventory_threshold)), 2)

# def calculate_reorder_quantity(stock_level, inventory_threshold, safety_stock=20):
#     return int(max(0, inventory_threshold - stock_level + safety_stock))

# def calculate_total_cost(order_quantity, cost_per_unit, delay_probability=0.0, delay_penalty_per_day=10):
#     return order_quantity * cost_per_unit + delay_probability * delay_penalty_per_day

# def generate_summary_supply(pred_df):
#     summaries = []
#     for _, row in pred_df.iterrows():
#         delay_risk = row.get('Delay_Probability', 0.0) * 100
#         stockout = row.get('Stockout_Probability', 0.0)
#         reorder_qty = int(row.get('Recommended_Reorder_Qty', 0))
#         total_cost = row.get('Total_Cost_Impact', 0.0)
#         order_id = int(row.get('Order_ID', -1))

#         delay_msg = (
#             "On track" if delay_risk < 10 else
#             "Moderate risk of delay" if delay_risk < 50 else
#             "High delay risk"
#         )
#         stock_msg = (
#             "Inventory sufficient" if stockout == 0 else
#             "Slight stock pressure" if stockout < 0.4 else
#             "Critical stock shortage"
#         )
#         cost_msg = (
#             "Minimal cost impact" if total_cost < 50000 else
#             "Moderate cost impact" if total_cost < 100000 else
#             "High cost impact"
#         )
#         summary = f"Order {order_id}: Delay {delay_msg}, Stock {stock_msg}, Cost {cost_msg}"
#         summaries.append(summary)
#     return summaries

# class SupplyChainModel:
#     def __init__(self):
#         self.delay_model = None
#         self.delay_days_model = None
#         self.encoder_supplier = LabelEncoder()
#         self.encoder_product = LabelEncoder()
#         self.load_models()

#     def preprocess(self, df, fit_encoders=False):
#         df = df.copy()
#         for c in ['Supplier', 'Product_SKU']:
#             if c not in df.columns:
#                 df[c] = "UNKNOWN"
#         if fit_encoders:
#             self.encoder_supplier.fit(df['Supplier'].astype(str))
#             self.encoder_product
