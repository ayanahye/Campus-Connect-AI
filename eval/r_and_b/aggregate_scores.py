import os
import pandas as pd

possible_f1_columns = ["BERT F1", "F1"]
other_metrics = ["METEOR", "ROUGE-L"]
standard_metrics = ["BERT F1", "METEOR", "ROUGE-L"]

unseen_scores = []
seen_scores = []

for filename in os.listdir("."):
    # req
    if not filename.endswith(".csv"):
        continue

    data = pd.read_csv(filename)

    f1_col = next((col for col in possible_f1_columns if col in data.columns), None)
    if f1_col is None or not all(m in data.columns for m in other_metrics):
        continue

    data = data.rename(columns={f1_col: "BERT F1"})
    data = data[standard_metrics].dropna()

    if "unseen" in filename.lower():
        unseen_scores.append(data)
    else:
        seen_scores.append(data)

unseen_avg = pd.concat(unseen_scores, ignore_index=True).mean() if unseen_scores else pd.Series({m: float('nan') for m in standard_metrics})
seen_avg = pd.concat(seen_scores, ignore_index=True).mean() if seen_scores else pd.Series({m: float('nan') for m in standard_metrics})

print("average scores for UNSEEN files")
print(unseen_avg.round(4))
#
print("\naverage scores for SEEN files")
print(seen_avg.round(4))

'''
average scores for UNSEEN files
BERT F1    0.8925
METEOR     0.3318
ROUGE-L    0.2750
dtype: float64

average scores for SEEN files
BERT F1    0.9057
METEOR     0.4170
ROUGE-L    0.3748

'''
