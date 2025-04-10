import json
import pandas as pd

with open('parks.json', 'r') as f:
    parks_data = json.load(f)

qa_pairs = []

for park in parks_data:
    park_name = park.get("name")
    street_number = park.get("streetnumber")
    street_name = park.get("streetname")
    neighbourhood_name = park.get("neighbourhoodname")
    hectare = park.get("hectare")
    facilities = park.get("facilities")
    washrooms = park.get("washrooms")
    advisories = park.get("advisories")
    special_features = park.get("specialfeatures")
    google_map_dest = park.get("googlemapdest", {})
    neighbourhood_url = park.get("neighbourhoodurl")

    if park_name and street_number and street_name:
        qa_pairs.append({
            "question": f"Where is {park_name} located?",
            "answer": f"{park_name} is located at {street_number} {street_name}."
        })
    if park_name and neighbourhood_name:
        qa_pairs.append({
            "question": f"In which neighbourhood is {park_name} located?",
            "answer": f"{park_name} is in the {neighbourhood_name} neighbourhood."
        })
    if park_name and hectare:
        qa_pairs.append({
            "question": f"How big is {park_name} in hectares?",
            "answer": f"{park_name} is {hectare} hectares in size."
        })
    if park_name and facilities:
        qa_pairs.append({
            "question": f"Does {park_name} have facilities?",
            "answer": f"{'Yes' if facilities == 'Y' else 'No'}, {park_name} has facilities."
        })
    if park_name and washrooms:
        qa_pairs.append({
            "question": f"Does {park_name} have washrooms?",
            "answer": f"{'Yes' if washrooms == 'Y' else 'No'}, {park_name} has washrooms."
        })
    if park_name and advisories:
        qa_pairs.append({
            "question": f"Are there any advisories for {park_name}?",
            "answer": f"{'Yes' if advisories == 'Y' else 'No'}, there are advisories for {park_name}."
        })
    if park_name and special_features:
        qa_pairs.append({
            "question": f"Does {park_name} have any special features?",
            "answer": f"{'Yes' if special_features == 'Y' else 'No'}, {park_name} has special features."
        })
    if park_name and neighbourhood_url:
        qa_pairs.append({
            "question": f"Where can I find more information about {neighbourhood_name}?",
            "answer": f"You can learn more about {neighbourhood_name} at: {neighbourhood_url}"
        })
    if park_name and google_map_dest:
        lat = google_map_dest.get("lat")
        lon = google_map_dest.get("lon")
        if lat and lon:
            qa_pairs.append({
                "question": f"Where is {park_name} located on the map?",
                "answer": f"{park_name} is located at latitude {lat} and longitude {lon}."
            })

df = pd.DataFrame(qa_pairs)
df.to_csv("park_qa_pairs.csv", index=False)







# might change qs