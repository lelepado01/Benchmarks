
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

MODEL = "SwinTransformerV2" #"MaskedAutoencoder"
dir_path = f"{MODEL}/"
METRIC = "gpu_power_usage_Context.TRAINING"

gpus = [8, 16, 32, 64, 128]
sizes = ["100M", "200M", "600M", "1B"]

def parse_int_list(ls):
    ls = ls[1:-1].split(',')
    return [int(element.strip()) for element in ls]

def parse_float_list(ls):
    ls = ls[1:-1].split(',')
    return [float(element.strip()) for element in ls]

def timestamp_to_minutes(ts):
    return ts / 60000

def timestamp_to_seconds(ts):
    return ts / 1000

def get_metric(data, metric):
    values = parse_float_list(data["entity"][metric]["prov-ml:metric_value_list"])
    times = parse_int_list(data["entity"][metric]["prov-ml:metric_timestamp_list"])
    
    return sum(values) / len(values), times

metrics = np.zeros((len(sizes), len(gpus)))
losses = np.zeros((len(sizes), len(gpus)))
times = np.zeros((len(sizes), len(gpus)))
for size in sizes:
    for gpu in gpus:
        file = dir_path + f"{MODEL}_{size}_{gpu}GPUS.json"
        try: 
            data = json.load(open(file))
        except:
            metrics[sizes.index(size), gpus.index(gpu)] = None
            losses[sizes.index(size), gpus.index(gpu)] = None
            times[sizes.index(size), gpus.index(gpu)] = None
            continue

        los, _ = get_metric(data, "train_loss_Context.TRAINING")
    
        m, time = get_metric(data, METRIC)
        time = [timestamp_to_seconds(ts) for ts in time]
        time = max(time) - min(time)

        metrics[sizes.index(size), gpus.index(gpu)] = m * gpu
        losses[sizes.index(size), gpus.index(gpu)] = los
        times[sizes.index(size), gpus.index(gpu)] = time

df_metrics = pd.DataFrame(metrics, columns=gpus, index=sizes)
df_losses = pd.DataFrame(losses, columns=gpus, index=sizes)
df_times = pd.DataFrame(times, columns=gpus, index=sizes)

# reverse the order of the rows
df_metrics = df_metrics.iloc[::-1]
df_losses = df_losses.iloc[::-1]
df_times = df_times.iloc[::-1]

# calculate relative time compared to 128 GPUs
time_128 = df_times[128]
df_times = df_times.transpose()
df_times = df_times / time_128
df_times = df_times.transpose()

df_total = df_metrics * df_losses * df_times

custom_cmap = sns.color_palette(palette='RdYlGn_r', as_cmap=True)
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 3, figsize=(25, 5))
sns.heatmap(df_metrics, annot=True, fmt=".2f", cmap=custom_cmap, ax=ax[0])
sns.heatmap(df_losses, annot=True, fmt=".2f", cmap=custom_cmap, ax=ax[1])
sns.heatmap(df_times, annot=True, fmt=".2f", cmap=custom_cmap, ax=ax[2])
ax[0].set_xlabel("Number of GPUs")
ax[0].set_ylabel("Model Size")
ax[0].set_title("Power * Number of GPUs")
ax[1].set_xlabel("Number of GPUs")
ax[1].set_ylabel("Model Size")
ax[1].set_title("Loss")
ax[2].set_xlabel("Number of GPUs")
ax[2].set_ylabel("Model Size")
ax[2].set_title("Relative Time")
# plt.savefig(f"{MODEL}_heatmaps.pdf")
plt.show()


