import csv
import re

with open('poorvi.txt', 'r', encoding='utf-8') as file:
    content = file.read()

entries = re.split(r"\nQues:\n?", content)
parsed_data = []

for entry in entries:
    if not entry.strip():
        continue
    ques_match = re.search(r'^(.*?)(?=Actual Answer:)', entry, re.DOTALL)
    actual_match = re.search(r'Actual Answer:\n?(.*?)(?=Gen Answer:)', entry, re.DOTALL)
    gen_match = re.search(r'Gen Answer:\n?(.*?)(?=Retrieved Docs:)', entry, re.DOTALL)
    docs_match = re.search(r'Retrieved Docs:\n?(.*?)(?=Bert Score:)', entry, re.DOTALL)
    score_match = re.search(r"'precision': \[(.*?)\], 'recall': \[(.*?)\], 'f1': \[(.*?)\]", entry)

    ques = ques_match.group(1).strip() if ques_match else ''
    actual = actual_match.group(1).strip() if actual_match else ''
    gen = gen_match.group(1).strip() if gen_match else ''
    docs = docs_match.group(1).strip() if docs_match else ''
    precision = score_match.group(1).strip() if score_match else ''
    recall = score_match.group(2).strip() if score_match else ''
    f1 = score_match.group(3).strip() if score_match else ''

    parsed_data.append([ques, actual, gen, docs, precision, recall, f1])

with open('poorvi-unseen-data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Ques', 'Actual_Answer', 'Gen_Answer', 'Retrieved_Docs', 'Bert_Precision', 'Bert_Recall', 'Bert_F1'])
    writer.writerows(parsed_data)
