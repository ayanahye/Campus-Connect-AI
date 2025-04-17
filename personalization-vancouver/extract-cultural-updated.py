import json
import pandas as pd

with open('cultural-spaces.json', 'r') as f:
    data = json.load(f)

qa_pairs = []

for entry in data:
    name = entry.get("cultural_space_name")
    address = entry.get("address")
    area = entry.get("local_area")
    space_type = entry.get("type")
    use = entry.get("primary_use")
    website = entry.get("website")
    ownership = entry.get("ownership")
    seats = entry.get("number_of_seats")
    active = entry.get("active_space")

    if not name or not address:
        continue

    answer = f"{name} is located at {address}."
    
    if area:
        answer += f" It is in the {area} area."
    if space_type:
        answer += f" It is a {space_type} primarily used for {use}."
    if ownership:
        answer += f" The space is owned by {ownership}."
    if seats:
        answer += f" It has {seats} seats."
    if active:
        answer += f" It is currently {'active' if active.lower() == 'yes' else 'inactive'}."
    if website:
        answer += f" You can find more information at: {website}."
    
    qa_pairs.append({
        "question": f"Tell me about {name}.",
        "answer": answer
    })


df = pd.DataFrame(qa_pairs)
df.to_csv("cultural_space_qa_pairs_up.csv", index=False)
