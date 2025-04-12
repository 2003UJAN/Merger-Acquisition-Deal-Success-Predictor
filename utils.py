import pandas as pd
import joblib

# Load trained model
def load_model(model_path="model/ma_model.pkl"):
    return joblib.load(model_path)

# Preprocess input data
def preprocess_input(acquirer_revenue, target_revenue, deal_value, industry_match, market_sentiment, prior_deals):
    industry_match_bin = 1 if industry_match.lower() in ['yes', 'y', '1', 'true'] else 0
    data = {
        "acquirer_revenue": [acquirer_revenue],
        "target_revenue": [target_revenue],
        "deal_value": [deal_value],
        "industry_match": [industry_match_bin],
        "market_sentiment": [market_sentiment],
        "prior_deals_acquirer": [prior_deals]
    }
    return pd.DataFrame(data)

# Predict success probability
def predict_success(model, input_df):
    probability = model.predict_proba(input_df)[0][1]
    return probability
