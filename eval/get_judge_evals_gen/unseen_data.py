import re

file_path = "gemini_unseen.txt"
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

Relevance: 4.74
Coherence: 4.65
Fluency: 4.96
Coverage: 3.91
Level of Detail: 2.91
Diversity: 1.26

'''