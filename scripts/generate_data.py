import os

import numpy as np
import pandas as pd

SEED = 7
N = 5000

DICTIONARY = """# Data dictionary: sales.csv
Columns:
- order_id: unique order id
- date: order date (2023-01-01 .. 2024-12-31)
- region: NA | EU | APAC | LATAM
- channel: online | retail | partner
- product: A | B | C | D | E
- units: integer units in the order (>= 1)
- unit_price: price per unit (USD)
- discount: fractional discount applied to the order (0..0.3)
- refunded: boolean; True if the order was later refunded

Metric definitions (ALWAYS use these):
- revenue = units * unit_price * (1 - discount)
- questions about revenue EXCLUDE refunded orders unless they explicitly say "including refunds"
- average order value = revenue / number of (non-refunded) orders
- discount cost = units * unit_price * discount
"""

def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    products = ["A", "B", "C", "D", "E"]
    prices = {"A": 25.0, "B": 150.0, "C": 80.0, "D": 45.0, "E": 300.0}

    product = rng.choice(products, N)
    df = pd.DataFrame({
        "order_id": [f"O{i:05d}" for i in range(N)],
        "date": pd.to_datetime(dates[rng.integers(0, len(dates), N)]),
        "region": rng.choice(["NA", "EU", "APAC", "LATAM"], N, p=[0.4, 0.3, 0.2, 0.1]),
        "channel": rng.choice(["online", "retail", "partner"], N, p=[0.5, 0.35, 0.15]),
        "product": product,
        "units": rng.integers(1, 20, N),
        "unit_price": np.round(np.array([prices[p] for p in product])
                               * rng.uniform(0.95, 1.05, N), 2),
        "discount": np.round(rng.uniform(0.0, 0.3, N), 2),
        "refunded": rng.random(N) < 0.06,
    })
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/sales.csv", index=False)
    with open("data/README.md", "w") as f:
        f.write(DICTIONARY)
    print(f"wrote data/sales.csv ({N} rows) + data/README.md")

if __name__ == "__main__":
    main()
