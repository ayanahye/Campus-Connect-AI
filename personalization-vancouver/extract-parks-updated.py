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
    neighbourhood_url = park.get("neighbourhoodurl")
    facilities = "Yes" if park.get("facilities") == "Y" else "No"
    washrooms = "Yes" if park.get("washrooms") == "Y" else "No"
    hectare = park.get("hectare")
    google_map_dest = park.get("googlemapdest")

    question = f"Where is {park_name} located, and does it have facilities and washrooms?"
    answer = (
        f"{park_name} is located at {street_number} {street_name}, in the {neighbourhood_name} area. "
        f"For more information, you can visit the neighbourhood page at {neighbourhood_url}. "
        f"The park is {hectare} hectares in size. It has facilities: {facilities}. Washrooms are available: {washrooms}. "
        f"You can view it on Google Maps at {google_map_dest['lat']}, {google_map_dest['lon']}."
    )

    qa_pairs.append({"question": question, "answer": answer})

qa_df = pd.DataFrame(qa_pairs)
qa_df.to_csv("park_qa_pairs-up.csv", index=False)
# p
print(qa_df.head())
