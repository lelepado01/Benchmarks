# Benchmarks

#### Files

- `3_heatmaps.py`: Generates heatmaps for the 3 versions: time, energy and loss.
- `gpu_power_heatmap.py`: Generates heatmaps for the GPU power consumption.
- `linecharts.py`: Generates linecharts for the 2 models, the metric can be changed.
- `loss_comparison.py`: Generates a linechart for comparing validation and train loss for the Masked Autoencoder (25 epoch run).

#### Data

- `data/`: Contains the data for the 25 epoch run of the Masked Autoencoder.
- `MaskeAutoencoder/`: Contains the data for the Masked Autoencoder runs, with the exception of the 8 GPU runs (larger than 100Mb).
- `SwinTransformerV2`: Contains the data for the Swin Transformer V2 runs.