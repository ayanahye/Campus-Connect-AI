import re
import statistics

with open("retrieval_seen.txt", "r", encoding="utf-8") as file:
    input_text = file.read()

accuracy_vals = [float(x) for x in re.findall(r"Accuracy,([0-9.]+)", input_text)]
precision_vals = [float(x) for x in re.findall(r"Precision,([0-9.]+)", input_text)]
recall_vals = [float(x) for x in re.findall(r"Recall,([0-9.]+)", input_text)]
mrr_vals = [float(x) for x in re.findall(r"MRR,([0-9.]+)", input_text)]
map_vals = [float(x) for x in re.findall(r"MAP,([0-9.]+)", input_text)]

def safe_avg(vals):
    return round(statistics.mean(vals), 3) if vals else 0.0

print(f"Accuracy: {safe_avg(accuracy_vals)}")
print(f"Precision: {safe_avg(precision_vals)}")
print(f"Recall: {safe_avg(recall_vals)}")
print(f"MRR: {safe_avg(mrr_vals)}")
print(f"MAP: {safe_avg(map_vals)}")

'''
Accuracy: 0.825
Precision: 0.804
Recall: 0.686
MRR: 0.941
MAP: 0.941
'''

# take w grain of salt, i didnt have much time to actually go thru the models outputs so not sure how good the evals are or even if the metrics should be used, in all maybe a better embedding model is needed but again we can highlight the tradeoff between time and accuracy