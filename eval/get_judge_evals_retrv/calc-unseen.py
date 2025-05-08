import re
import statistics

with open("retrieval_unseen.txt", "r", encoding="utf-8") as file:
    text = file.read()

accuracy = [float(x) for x in re.findall(r"Accuracy,([0-9.]+)", text)]
precision = [float(x) for x in re.findall(r"Precision,([0-9.]+)", text)]
recall = [float(x) for x in re.findall(r"Recall,([0-9.]+)", text)]
mrr = [float(x) for x in re.findall(r"MRR,([0-9.]+)", text)]
map_ = [float(x) for x in re.findall(r"MAP,([0-9.]+)", text)]

def avg(lst):
    return round(statistics.mean(lst), 3) if lst else 0.0

print(f"Accuracy: {avg(accuracy)}")
print(f"Precision: {avg(precision)}")
print(f"Recall: {avg(recall)}")
print(f"MRR: {avg(mrr)}")
print(f"MAP: {avg(map_)}")

'''
Accuracy: 0.319
Precision: 0.319
Recall: 0.194
MRR: 0.417
MAP: 0.361
'''

# take w grain of salt, i didnt have much time to actually go thru the models outputs so not sure how good the evals are or even if the metrics should be used, in all maybe a better embedding model is needed but again we can highlight the tradeoff between time and accuracy