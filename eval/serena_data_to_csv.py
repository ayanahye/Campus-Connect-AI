import csv
import re

input_file = "serena.txt"
output_file = "serena-unseen-data.csv"

with open(input_file, "r", encoding="utf-8") as file:
    text = file.read()

entries = re.split(r"={10,}\nBenchmark Query \d+: ", text)[1:]

data = []
for entry in entries:
    parts = entry.strip().split("\n", 1)
    question = parts[0].strip()
    rag_match = re.search(r"RAG Response:\s+(.*?)\n\n", entry, re.DOTALL)
    rag = rag_match.group(1).strip() if rag_match else ""

    bench_match = re.search(r"\(Benchmark\) Answer:\s+(.*?)\nBERT Score", entry, re.DOTALL)
    bench_answer = bench_match.group(1).strip() if bench_match else ""

    score_match = re.search(r"BERT Score ==.*?precision': \[(.*?)\], 'recall': \[(.*?)\], 'f1': \[(.*?)\]", entry)
    precision, recall, f1 = score_match.groups() if score_match else ("", "", "")

    data.append({
        "Question": question,
        "RAG Response": rag,
        "Benchmark Answer": bench_answer,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["Question", "RAG Response", "Benchmark Answer", "Precision", "Recall", "F1"])
    writer.writeheader()
    for row in data:
        writer.writerow(row)

