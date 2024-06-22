
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

MODEL = "MaskedAutoencoder" #"SwinTransformerV2"
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
    return sum(values) / len(values)

metrics = np.zeros((len(sizes), len(gpus)))
for size in sizes:
    for gpu in gpus:
        file = dir_path + f"MaskedAutoencoder_{size}_{gpu}GPUS.json"
        try: 
            data = json.load(open(file))
        except:
            metrics[sizes.index(size), gpus.index(gpu)] = None
            continue
    
        m = get_metric(data, METRIC)
        metrics[sizes.index(size), gpus.index(gpu)] = m

df_metrics = pd.DataFrame(metrics, columns=gpus, index=sizes)
df_metrics = df_metrics.iloc[::-1]

custom_cmap = sns.color_palette(palette='RdYlGn_r', as_cmap=True)
sns.set_theme()
sns.set_context("paper")
sns.heatmap(df_metrics, annot=True, fmt=".2f", cmap=custom_cmap)
plt.xlabel("Number of GPUs")
plt.ylabel("Model Size")
plt.title(f"{MODEL} GPU Power Usage (W)")
# plt.savefig("gpu_power.pdf")
plt.show()

