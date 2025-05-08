import csv
import re

with open('ayana.txt', 'r', encoding='utf-8') as f:
    data = f.read()

blocks = re.split(r'\n(?=Ques:)', data)

with open('ayana-unseen-data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        'Question', 'Actual Answer', 'Generated Answer',
        'Retrieved Docs', 'BERT Precision', 'BERT Recall', 'BERT F1'
    ])
    writer.writeheader()

    for block in blocks:
        question = re.search(r'Ques:\s*(.*?)\n', block, re.DOTALL)
        actual = re.search(r'(?:Actual Answer|Actual Ans):\s*(.*?)\n(?:Generated Answer|Gen Answer|Gen Ans):', block, re.DOTALL)
        generated = re.search(r'(?:Generated Answer|Gen Answer|Gen Ans):\s*(.*?)\nRetrieved Docs:', block, re.DOTALL)
        retrieved = re.search(r'Retrieved Docs:\s*(.*?)\nBert Score:', block, re.DOTALL)
        bert = re.search(r"Bert Score: 'precision': \[(.*?)\], 'recall': \[(.*?)\], 'f1': \[(.*?)\]", block)

        writer.writerow({
            'Question': question.group(1).strip() if question else '',
            'Actual Answer': actual.group(1).strip() if actual else '',
            'Generated Answer': generated.group(1).strip() if generated else '',
            'Retrieved Docs': retrieved.group(1).strip() if retrieved else '',
            'BERT Precision': bert.group(1).strip() if bert else '',
            'BERT Recall': bert.group(2).strip() if bert else '',
            'BERT F1': bert.group(3).strip() if bert else '',
        })
