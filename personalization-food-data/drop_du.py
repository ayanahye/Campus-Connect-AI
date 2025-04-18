import pandas as pd

df = pd.read_csv("qa_pairs_food.csv")

df=df.drop_duplicates()

df.to_csv("qa_pairs_food.csv", index=False)