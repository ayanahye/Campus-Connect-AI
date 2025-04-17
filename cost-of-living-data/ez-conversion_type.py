import pandas as pd

df = pd.read_csv("original_2021_bc_cost_by_household_type.csv")
df = df.dropna()
df.to_csv("no_na_original_bc_cost_ht.csv", index=False)

qa_pairs = []

for index, row in df.iterrows():
    question = f"What was the average household expenditure on {row['Household expenditures, summary-level categories']} for {row['Household type']} in 2021?"
    answer = f"${row['VALUE']}"
    qa_pairs.append([question, answer])

qa_df = pd.DataFrame(qa_pairs, columns=["question", "answer"])
qa_df = qa_df.dropna()
qa_df.to_csv("qa_pairs_household_type.csv", index=False)
