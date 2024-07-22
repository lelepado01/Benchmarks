
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tak_dir = "tak_data/"
TOP = 1
MODEL = "SwinTransformerV2" #"MaskedAutoencoder" #
if MODEL == "MaskedAutoencoder":
    file = f"mae_acc{TOP}.csv"
elif MODEL == "SwinTransformerV2":
    file = f"swt_acc{TOP}.csv"

df = pd.read_csv(tak_dir + file)
df = df.set_index("nparams")

if TOP == 1:
    min_value = 39.601850509643555
    max_value = 53.51542830467224
elif TOP == 5:
    min_value = 85.60185432434082
    max_value = 96.3009238243103

sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2.0)
plt.figure(figsize=(12, 8))
sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=min_value, vmax=max_value)
plt.xlabel("Number of GPUs")
plt.ylabel("Model Size")
plt.title(f"{MODEL} Top@{TOP}")
plt.savefig(f"{MODEL}_TOP@{TOP}_tak.pdf")
plt.show()
