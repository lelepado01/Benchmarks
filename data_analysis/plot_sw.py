
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/SW_loss_square.csv")
df = pd.read_csv("/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/MAE_val_loss_cleaned_square.csv")
df = df.set_index("Unnamed: 0")
df = df.iloc[:3]
df = df.transpose()

print(df)

# Plot as bar
df.plot(kind="bar", stacked=False)
plt.xticks(range(len(df.index)), df.index)
plt.xlabel("GPUs")
plt.ylabel("Test loss")
plt.legend(title="Model size (MAE)")
plt.title("Test loss for different model sizes")
plt.tight_layout()
plt.savefig("/Users/gabrielepadovani/Desktop/Università/PhD/data_analysis/imgs/MAE_loss.pdf")
plt.show()
