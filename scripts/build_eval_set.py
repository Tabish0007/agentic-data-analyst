import json
import os

import pandas as pd

df = pd.read_csv("data/sales.csv", parse_dates=["date"])
df["revenue"] = df["units"] * df["unit_price"] * (1 - df["discount"])
df["discount_cost"] = df["units"] * df["unit_price"] * df["discount"]
df["year"] = df["date"].dt.year
df["qtr"] = df["date"].dt.quarter
df["month"] = df["date"].dt.month

nr = df[~df["refunded"]]
r23, r24 = nr[nr["year"] == 2023], nr[nr["year"] == 2024]
r3_all, r4_all = df[df["year"] == 2023], df[df["year"] == 2024]

tasks = []
def add(qid, question, answer, complexity):
    tasks.append({"id": qid, "question": question,
                  "answer": answer, "complexity": complexity})

add("t01", "What was total revenue in 2024?", round(float(r24["revenue"].sum()), 2), "simple")
add("t02", "Which region had the highest total revenue in 2023?",
    nr[nr["year"] == 2023].groupby("region")["revenue"].sum().idxmax(), "simple")
eu_q2 = nr[(nr["region"] == "EU") & (nr["year"] == 2024) & (nr["qtr"] == 2)]
add("t03", "What was the average order value in the EU in Q2 2024?",
    round(float(eu_q2["revenue"].sum() / len(eu_q2)), 2), "simple")
add("t04", "How many orders in the partner channel in 2024 were refunded?",
    int(r4_all[r4_all["channel"] == "partner"]["refunded"].sum()), "simple")
add("t05", "Which product had the highest total units sold in 2024 (excluding refunded)?",
    r24.groupby("product")["units"].sum().idxmax(), "simple")
add("t06", "What percentage of 2024 orders were refunded (rounded to 1 decimal)?",
    round(float(100 * r4_all["refunded"].mean()), 1), "simple")
add("t07", "By what percentage did total revenue grow from 2023 to 2024 (rounded to 1 decimal)?",
    round(float(100 * (r24["revenue"].sum() / r23["revenue"].sum() - 1)), 1), "complex")
add("t08", "What was the median unit price across all non-refunded orders in 2024?",
    round(float(r24["unit_price"].median()), 2), "simple")
m = (r24.groupby(["month", "channel"])["revenue"].sum().unstack()
       .reindex(columns=["online", "retail"], fill_value=0))
add("t09", "In how many months of 2024 did online-channel revenue exceed retail-channel revenue (non-refunded)?",
    int((m["online"] > m["retail"]).sum()), "complex")
add("t10", "What was the total discount cost in LATAM in 2024, counting all orders including refunded?",
    round(float(r4_all[r4_all["region"] == "LATAM"]["discount_cost"].sum()), 2), "simple")
add("t11", "What was the total number of orders in 2023?", int(len(r3_all)), "simple")
add("t12", "Which product generated the most revenue in APAC in 2024 (non-refunded)?",
    nr[(nr["region"] == "APAC") & (nr["year"] == 2024)].groupby("product")["revenue"].sum().idxmax(),
    "simple")

os.makedirs("evals/tasks", exist_ok=True)
with open("evals/tasks/tasks.jsonl", "w") as f:
    for t in tasks:
        f.write(json.dumps(t) + "\n")
print(f"wrote evals/tasks/tasks.jsonl ({len(tasks)} tasks)")
