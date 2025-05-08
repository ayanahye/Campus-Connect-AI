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

'''
actually highlights 2 things (given that my code is correct: please verify)
- 1) potentially we are missing documents in relevant areas (here is another key, we can always add more data and while testing and searching online for relevant faq from international students we realized we are missing some important areas (these results also support that) thus highlighting the importance of developing alongside real users i.e. international students but due to time constraints we leave for future iterations on this)
- 2) potentially that the generator can do well enough given its high scores on generation on unseen data even with docs that arent so helpful. Also, that the generator model was able to ignore info when it was not relveant

we can report this in presentation
'''