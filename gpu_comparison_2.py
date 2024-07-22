
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


MODEL = "MaskedAutoencoder" #"SwinTransformerV2" #

dir_path = f"{MODEL}/"
METRIC = "train_loss_Context.TRAINING"

PARAMS = "200M"
GPUS = [8, 16, 32, 64, 128]

sns.set(style="whitegrid")
sns.set_context("paper")
sns.set(font_scale=1.2)
plt.figure(figsize=(12, 8))

for gpu in GPUS:
    file = dir_path + f"{MODEL}_{PARAMS}_{gpu}GPUS.json"
    try:
        data = json.load(open(file))
    except:
        continue
    df = get_metric(data, METRIC)
    # print(len(df["value"]))

    # smooth the curve
    df["value"] = df["value"].rolling(window=30).mean()

    plt.plot(df["time"], df["value"], label=f"{gpu} GPUs")
    # plt.plot(range(len(df["value"])), df["value"], label=f"{param}")

# zoom to area below 0.5
plt.ylim(0, 0.075)

plt.legend()
plt.xlabel("Training Time (minutes)")
plt.ylabel("Training Loss")
plt.title(f"{MODEL} {PARAMS}")
plt.tight_layout()
plt.savefig(f"{MODEL}_{PARAMS}_loss_comparison.pdf")
plt.show()






