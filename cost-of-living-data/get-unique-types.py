import pandas as pd

df = pd.read_csv("original_2021_bc_cost_by_household_type.csv")
unique_values = df["Household expenditures, summary-level categories"].unique()
for value in unique_values:
    print(value)
print(len(unique_values))
