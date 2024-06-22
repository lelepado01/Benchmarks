import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

MODEL = "MaskedAutoencoder" #"SwinTransformerV2"
GPUS = 128

dir_path = f"{MODEL}/"
METRIC = "gpu_power_usage_Context.TRAINING"

def parse_int_list(ls):
    ls = ls[1:-1].split(',')
    return [int(element.strip()) for element in ls]

def parse_float_list(ls):
    ls = ls[1:-1].split(',')
    return [float(element.strip()) for element in ls]

def timestamp_to_minutes(ts):
    return ts / 60000

def get_metric(data, metric):
    epochs = parse_int_list(data["entity"][metric]["prov-ml:metric_epoch_list"])
    values = parse_float_list(data["entity"][metric]["prov-ml:metric_value_list"])
    times = parse_int_list(data["entity"][metric]["prov-ml:metric_timestamp_list"])
    
    # convert to minutes and sort
    times = [timestamp_to_minutes(ts-times[0]) for ts in times]
    df = pd.DataFrame({"epoch": epochs, "value": values, "time": times}).drop_duplicates()#.diff().fillna(0)
    df = df.sort_values(by="time")
    return df


### GPU POWER USAGE ###
file = dir_path + f"{MODEL}_1B_{GPUS}GPUS.json"
data1 = json.load(open(file))
df1 = get_metric(data1, METRIC)

file2 = dir_path + f"{MODEL}_600M_{GPUS}GPUS.json"
data2 = json.load(open(file2))
df2 = get_metric(data2, METRIC)

file3 = dir_path + f"{MODEL}_200M_{GPUS}GPUS.json"
data3 = json.load(open(file3))
df3 = get_metric(data3, METRIC)

file4 = dir_path + f"{MODEL}_100M_{GPUS}GPUS.json"
data4 = json.load(open(file4))
df4 = get_metric(data4, METRIC)

plt.figure(figsize=(10, 6))
sns.lineplot(x=range(len(df1)), y="value", data=df1, label="1B")
sns.lineplot(x=range(len(df2)), y="value", data=df2, label="600M")
sns.lineplot(x=range(len(df3)), y="value", data=df3, label="200M")
sns.lineplot(x=range(len(df4)), y="value", data=df4, label="100M")
plt.xlabel("Execution Steps")
plt.ylabel("GPU Power Usage (W)")
plt.title(f"{MODEL} {GPUS} GPUs")
plt.legend()
# plt.savefig("power_usage.pdf")
plt.show()
#######################