
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

def parse_int_list(ls):
    ls = ls[1:-1].split(',')
    return [int(element.strip()) for element in ls]

def parse_float_list(ls):
    ls = ls[1:-1].split(',')
    return [float(element.strip()) for element in ls]

def timestamp_to_minutes(ts):
    return ts / 60000

def get_metrics(data):
    return data["entity"].keys()

def get_metric(data, metric):
    epochs = parse_int_list(data["entity"][metric]["prov-ml:metric_epoch_list"])
    values = parse_float_list(data["entity"][metric]["prov-ml:metric_value_list"])
    times = parse_int_list(data["entity"][metric]["prov-ml:metric_timestamp_list"])
    # convert to minutes and sort
    times = [timestamp_to_minutes(ts-times[0]) for ts in times]
    df = pd.DataFrame({"epoch": epochs, "value": values, "time": times}).drop_duplicates()#.diff().fillna(0)

    df = df.sort_values(by="time")
    return df


MODEL =  "MaskedAutoencoder" #"SwinTransformerV2" #

dir_path = f"{MODEL}/"
METRIC = "train_loss_Context.TRAINING"

PARAMS = ["100M", "200M", "600M", "1B"]
GPUS = [8, 16, 32, 64, 128]

dfs = []
for param in PARAMS:
    for gpu in GPUS:
        file = dir_path + f"{MODEL}_{param}_{gpu}GPUS.json"
        try:
            data = json.load(open(file))
        except:
            dfs.append([None, param, gpu])
            continue
        df = get_metric(data, METRIC)
        df1_final = df["value"].iloc[-1]
        dfs.append([df1_final, param, gpu])

# create a dataframe
df = pd.DataFrame(dfs, columns=["value", "param", "gpu"])


# create a pivot table
pivot = df.pivot(index="param", columns="gpu", values="value")

print(pivot)

# sort based on the number of parameters
PARAMS.reverse()
pivot = pivot.reindex(PARAMS)

print("MIN: ", pivot.min().min())
print("MAX: ", pivot.max().max())

min_scale = 0.006
max_scale = 0.02369

# plot heatmap
sns.set(font_scale=2)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", vmax=max_scale, vmin=min_scale)
plt.title("Training Loss (MaskedAutoencoder)")
plt.xlabel("GPUs")
plt.ylabel("Model Parameters")
plt.tight_layout()
plt.savefig(f"{MODEL}_GPU_Comparison.pdf")
plt.show()