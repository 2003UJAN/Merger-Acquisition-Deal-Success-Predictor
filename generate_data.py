import pandas as pd
import numpy as np

def generate_ma_data(num_samples=1000, output_path='data/ma_data.csv'):
    np.random.seed(42)

    # Generate random financials and categorical features
    data = {
        "acquirer_revenue": np.random.uniform(100, 5000, size=num_samples),
        "target_revenue": np.random.uniform(50, 3000, size=num_samples),
        "deal_value": np.random.uniform(10, 8000, size=num_samples),
        "industry_match": np.random.choice([0, 1], size=num_samples),  # 0: no, 1: yes
        "market_sentiment": np.random.uniform(-1, 1, size=num_samples),
        "prior_deals_acquirer": np.random.randint(0, 30, size=num_samples),
    }

    df = pd.DataFrame(data)

    # Generate synthetic "success" outcome (binary target)
    df["success"] = (
        (df["industry_match"] == 1).astype(int)
        + (df["market_sentiment"] > 0).astype(int)
        + (df["deal_value"] < 5000).astype(int)
        + (df["prior_deals_acquirer"] > 5).astype(int)
    )
    df["success"] = (df["success"] >= 3).astype(int)

    df.to_csv(output_path, index=False)
    print(f"✅ Synthetic M&A dataset saved to {output_path}")

if __name__ == "__main__":
    generate_ma_data()
