import pandas as pd

file = "Hike Database - Best Hikes BC.xlsx"
df = pd.read_excel(file, header=1)


df.columns = df.columns.str.strip()
# print("Columns:", df.columns.tolist())

qa_data = []
for _, row in df.iterrows():
    question = f"Can you tell me about the {row['HIKE NAME']} hike?"
    answer = (
        f"Ranking: {row['RANKING']}\n"
        f"Difficulty: {row['DIFFICULTY']}\n"
        f"Distance: {row['DISTANCE (KM)']} km\n"
        f"Elevation Gain: {row['ELEVATION GAIN (M)']} m\n"
        f"Gradient: {row['GRADIENT']}\n"
        f"Time: {row['TIME (HOURS)']} hours\n"
        f"Dogs Allowed: {row['DOGS']}\n"
        f"4x4 Needed: {row['4X4']}\n"
        f"Season: {row['SEASON']}\n"
        f"Region: {row['REGION']}"
    )
    qa_data.append({"Question": question, "Answer": answer})

output_path = "hikes_qa.csv"
qa_df = pd.DataFrame(qa_data)
qa_df.to_csv(output_path, index=False)
