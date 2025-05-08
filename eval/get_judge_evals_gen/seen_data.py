import re

file_path = "gemini_seen.txt"
scores = {
    "Relevance": [],
    "Coherence": [],
    "Fluency": [],
    "Coverage": [],
    "Level of Detail": [],
    "Diversity": []
}

with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

entries = content.strip().split('---')
for entry in entries:
    if "Gemini Evaluation:" in entry:
        match = re.search(r"Gemini Evaluation:(.*)", entry, re.DOTALL)
        if match:
            eval_text = match.group(1).strip()
            pattern = r"(Relevance|Coherence|Fluency|Coverage|Level of Detail|Diversity) (\d),"
            matches = re.findall(pattern, eval_text)
            for category, score in matches:
                scores[category].append(int(score))

for category, vals in scores.items():
    avg = sum(vals) / len(vals) if vals else 0
    print(f"{category}: {avg:.2f}")

'''

Relevance: 4.92
Coherence: 4.81
Fluency: 5.00
Coverage: 3.96
Level of Detail: 2.92
Diversity: 1.19

'''
