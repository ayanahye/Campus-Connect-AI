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
    sqft = entry.get("square_feet")
    seats = entry.get("number_of_seats")
    active = entry.get("active_space")
    if name and address:
        qa_pairs.append({
            "question": f"Where is {name} located?",
            "answer": f"{name} is located at {address}."
        })
    if name and area:
        qa_pairs.append({
            "question": f"In which local area is {name} located?",
            "answer": f"{name} is in the {area} neighborhood."
        })
    if name and space_type:
        qa_pairs.append({
            "question": f"What type of cultural space is {name}?",
            "answer": f"{name} is categorized as a {space_type}."
        })
    if name and use:
        qa_pairs.append({
            "question": f"What is the primary use of {name}?",
            "answer": f"The primary use of {name} is {use}."
        })
    if name and ownership:
        qa_pairs.append({
            "question": f"Who owns {name}?",
            "answer": f"{name} is owned by {ownership}."
        })
    if name and sqft:
        qa_pairs.append({
            "question": f"How big is {name} in square feet?",
            "answer": f"{name} has approximately {sqft} square feet of space."
        })
    if name and seats:
        qa_pairs.append({
            "question": f"How many seats are available at {name}?",
            "answer": f"{name} has {seats} seats."
        })
    if name and website:
        qa_pairs.append({
            "question": f"Where can I find more information about {name}?",
            "answer": f"You can learn more at: {website}"
        })
    if name and active:
        qa_pairs.append({
            "question": f"Is {name} currently active?",
            "answer": f"Yes, {name} is currently active." if active.lower() == "yes" else f"No, {name} is not currently active."
        })

df = pd.DataFrame(qa_pairs)
df.to_csv("cultural_space_qa_pairs.csv", index=False)

