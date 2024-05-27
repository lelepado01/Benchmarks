
import pandas as pd 

path1 = "/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/wandb_export_2024-05-27T18 53 30.129+02 00.csv"

df = pd.read_csv(path1)

cols = [col for col in df.columns if "step" not in col and "runtime" not in col and  "MIN" not in col and "MAX" not in col]

df = df[cols]

df.to_csv("/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/MAE_test.csv", index=False)