import pandas as pd

df = pd.read_csv("no_na_original_bc_cost_ht.csv")
# avg month, year, grouping was not effective on model
df["VALUE"] = df["VALUE"].astype(float)
qa_pairs = []

for index, row in df.iterrows():
    category = row['Household expenditures, summary-level categories']
    household = row['Household type']
    value = row['VALUE']
    year = row['REF_DATE']
    monthly_value = value / 12
    monthly_str = f"${monthly_value:,.2f}"
    yearly_str = f"${value:,.2f}"

    question = f"What was the average household expenditure on {category} for {household} in {year}?"
    answer = f"The yearly average was {yearly_str}, which is about {monthly_str} per month."
    
    qa_pairs.append([question, answer])

qa_df = pd.DataFrame(qa_pairs, columns=["question", "answer"])
qa_df.to_csv("qa_pairs_year_and_month_avg.csv", index=False, quoting=1)
