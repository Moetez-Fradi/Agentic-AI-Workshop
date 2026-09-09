"""Generates data/marketing_sales.csv — a synthetic, messy-ish sales log.

Run once: python generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(7)

REGIONS = ["EMEA", "NA", "APAC"]
PRODUCTS = {
    "Product A": 500,
    "Product B": 350,
    "Product C": 900,
    "Product D": 150,
    "Product E": 600,
    "Product F": 250,
}
DAYS = 90
START = "2026-05-01"

# Products that quietly decline in EMEA specifically, so the
# "top 3 underperforming in EMEA" question has a real answer.
EMEA_DECLINERS = {"Product B", "Product D", "Product F"}

dates = pd.date_range(START, periods=DAYS, freq="D")
rows = []

for region in REGIONS:
    for product, base_revenue in PRODUCTS.items():
        trend = 0.0
        declining = region == "EMEA" and product in EMEA_DECLINERS
        for day_idx, date in enumerate(dates):
            if declining:
                trend -= np.random.uniform(1.5, 4.0)  # steady erosion
            else:
                trend += np.random.uniform(-1.0, 1.5)  # roughly flat/growing

            noise = np.random.normal(0, base_revenue * 0.08)
            revenue = max(0, base_revenue + trend + noise)
            spend = max(0, base_revenue * 0.2 + np.random.normal(0, base_revenue * 0.03))
            units = max(0, int(revenue / np.random.uniform(8, 15)))

            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "region": region,
                    "product": product,
                    "revenue": round(revenue, 2),
                    "marketing_spend": round(spend, 2),
                    "units_sold": units,
                }
            )

df = pd.DataFrame(rows)
df.to_csv("data/marketing_sales.csv", index=False)
print(f"Wrote data/marketing_sales.csv with {len(df)} rows")
