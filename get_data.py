import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

MODEL = "MaskedAutoencoder" #"SwinTransformerV2"

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
    
    times = [(ts-times[0]) for ts in times]
    df = pd.DataFrame({"epoch": epochs, "value": values, "time": times}).drop_duplicates()#.diff().fillna(0)
    df = df.sort_values(by="time")
    return df

GPUS = [8, 16, 32, 64, 128]

gpup = []
time_s = []
time_ms = []
test_loss = []
for gpu in GPUS:
    file = dir_path + f"{MODEL}_1B_{gpu}GPUS.json"
    data = json.load(open(file))
    df = get_metric(data, "gpu_power_usage_Context.TRAINING")
    time_s.append(round(df["time"].iloc[-1] / 1000, 2))
    time_ms.append(round(df["time"].iloc[-1], 2))
    gpup.append(round(df["value"].mean(), 2))
    df = get_metric(data, "test_loss_Context.EVALUATION")
    test_loss.append(round(df["value"].mean(), 6))

print(test_loss)

