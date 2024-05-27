
import pandas as pd

path1 = "/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/MAE_test.csv"

indexes = ["100M", "600M", "1B"]
cols = ["8", "16", "32", "64", "128"]

square_df = pd.DataFrame({
    "100M": [0.0, 0.0, 0.0, 0.0, 0.0],
    "600M": [0.0, 0.0, 0.0, 0.0, 0.0],
    "1B": [0.0, 0.0, 0.0, 0.0, 0.0]
}).transpose()
square_df.columns = cols

df = pd.read_csv(path1)
# df.drop(columns=["Step"], inplace=True)
# df.drop(columns=["epoch"], inplace=True)
df.drop(columns=["Relative Time (Wall)"], inplace=True)

for i in indexes: 
    for c in cols: 
        ls_values = []
        for column in df.columns:
            gpus = column.split("_")[2][:3].replace("G", "").replace("P", "").strip()
            if i in column and c == gpus and "SW" in column:
                val = df[column].item()
                ls_values.append(val)

        square_df.loc[i, c] = sum(ls_values) / len(ls_values) if len(ls_values) > 0 else 0.0

print(square_df)

square_df.to_csv("/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/MAE_test_square.csv", index=True)