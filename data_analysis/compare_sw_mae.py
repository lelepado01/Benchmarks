
import pandas as pd
import matplotlib.pyplot as plt

path1 = "/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/MAE_val_loss_cleaned_square.csv"
path2 = "/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/SW_loss_square.csv"

df_mae = pd.read_csv(path1)
df_mae = df_mae.set_index("Unnamed: 0")
df_mae = df_mae.iloc[:2]
# df_mae = df_mae.transpose()

df_sw = pd.read_csv(path2)
df_sw = df_sw.set_index("Unnamed: 0")
df_sw = df_sw.iloc[:2]
# df_sw = df_sw.transpose()

# sum over gpu models
df_mae["MAE loss"] = df_mae.sum(axis=1) / 5
df_sw["SW loss"] = df_sw.sum(axis=1) / 5

df_mae = df_mae["MAE loss"]
df_sw = df_sw["SW loss"]

df_join = pd.concat([df_mae, df_sw], axis=1)    
print(df_join)

df_join.plot(kind="bar", stacked=False)
plt.xlabel("Model size")
plt.ylabel("Test loss")
plt.title("Test loss for different models and sizes")
plt.tight_layout()
plt.savefig("/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/imgs/loss_comparison.pdf")
plt.show()

# join the two dataframes
# df = pd.concat([df_mae, df_sw], axis=1)
# df.columns = ["MAE 100M", "MAE 600M", "SW 100M", "SW 600M"]
# # arrange columns
# df = df[["MAE 100M", "SW 100M", "MAE 600M", "SW 600M"]]
# print(df)

# # Plot as bar
# df.plot(kind="bar", stacked=False)
# plt.xlabel("GPUs")
# plt.ylabel("Test loss")
# plt.title("Test loss for different models and sizes")
# plt.legend(title="Model + size configuration")
# plt.tight_layout()
# plt.savefig("/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/imgs/loss_comparison_split.pdf")
# plt.show()




