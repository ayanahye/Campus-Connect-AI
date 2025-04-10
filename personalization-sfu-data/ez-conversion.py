import re
import pandas as pd

with open('raw.txt', 'r', encoding='utf-8', errors='ignore') as file:
    raw_data = file.read()

pattern = re.compile(r'<b><a href="/clubs/\d+/info">([^<]+)</a></b><br>([^<]+)')

matches = re.findall(pattern, raw_data)

qa_pairs = []

for club_name, description in matches:
    cleaned_description = ' '.join(description.splitlines()).strip()
    
    question = f"What is the description of the club {club_name} at SFU?"
    answer = cleaned_description
    qa_pairs.append([question, answer])

qa_df = pd.DataFrame(qa_pairs, columns=["question", "answer"])
qa_df.to_csv("qa_pairs_sfu_clubs.csv", index=False)




