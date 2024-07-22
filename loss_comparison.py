
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

import json
data = json.load(open("data/full_run.json"))

# print(get_metrics(data))

data1 = get_metric(data, "train_loss_Context.TRAINING")
data2 = get_metric(data, "val_loss_Context.VALIDATION")

data1 = data1.sort_values(by="time")
data2 = data2.sort_values(by="time")

data1["time"] = data1["time"] - data1["time"].iloc[0]
data2["time"] = data2["time"] - data2["time"].iloc[0]

data1 = data1[data1["epoch"] < 12]
data2 = data2[data2["epoch"] < 12]

### VALIDATION + TRAINING LOSS per EPOCH ###
data1 = data1.groupby("epoch").mean().reset_index()
data2 = data2.groupby("epoch").mean().reset_index()
sns.set(style="whitegrid")
sns.set(font_scale=2.0)
plt.figure(figsize=(10, 6))
plt.plot(data1["epoch"], data1["value"], label="Train Loss")
plt.plot(data2["epoch"], data2["value"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_comparison_epoch.pdf")
plt.show()
#############################################

### VALIDATION + TRAINING LOSS per TIMESTEP ###
# data1 = data1[data1["time"] < 0.9e6]
# data2 = data2[data2["time"] < 0.9e6]
# sns.set(style="whitegrid")
# plt.figure(figsize=(10, 6))
# plt.plot(data1["time"], data1["value"], label="Train Loss")
# plt.plot(data2["time"], data2["value"], label="Validation Loss")
# plt.xlabel("Timestep")
# plt.ylabel("Loss")
# plt.legend()
# # plt.savefig("loss_comparison_timestep.pdf")
# plt.show()
#############################################