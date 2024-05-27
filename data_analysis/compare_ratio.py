
import pandas as pd
import matplotlib.pyplot as plt

path_loss = "/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/SW_loss_square.csv"
# path_loss = "/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/MAE_loss_cleaned_square.csv"
path_energy = "/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/SW_power_square.csv"
# path_energy = "/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/MAE_power_cleaned_square.csv"

path_time = "/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/SW_execution_time.csv"
# path_time = "/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/MAE_execution_time.csv"


df_loss = pd.read_csv(path_loss)
df_loss = df_loss.set_index("Unnamed: 0")

df_power = pd.read_csv(path_energy)
df_power = df_power.set_index("Unnamed: 0")

df_time = pd.read_csv(path_time)
df_time = df_time.set_index("Unnamed: 0")

# calculate energy consumption total = power * time * number of gpus
ngpus = df_power.columns.astype(int)
df_energy = df_power * df_time
df_energy = df_energy * ngpus

# # normalize
# df_power = df_power / (df_power.max() - df_power.min())
# df_loss = df_loss / (df_loss.max() - df_loss.min())
# df_energy = df_energy / (df_energy.max() - df_energy.min())

# # find best ratio
# df = df_loss / df_energy
df = df_loss * df_energy
print(df)

# Plot as heatmap
plt.imshow(df, interpolation='nearest', cmap="YlOrRd", aspect='auto')
plt.xticks(range(len(df.columns)), df.columns)
plt.yticks(range(len(df.index)), df.index)
plt.xlabel("GPUs")
plt.ylabel("Model size (SW)")
plt.title("Test loss x energy consumption (lower is better)")
plt.colorbar()
plt.tight_layout()
plt.savefig("/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/imgs/SW_loss_energy.pdf")
plt.show()

# Plot as scatter
# df = df.stack().reset_index()
# df.columns = ["Model size", "GPU", "Ratio"]
# print(df)

# plt.scatter(df["GPU"], df["Model size"], sizes=df["Ratio"] * 100, c=df["Ratio"], cmap="YlOrRd")
# plt.xlabel("Model size")
# plt.ylabel("Ratio")
# plt.title("Ratio between test loss and energy consumption")
# plt.tight_layout()
# plt.savefig("/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/imgs/loss_energy_scatter.pdf")
# plt.show()

