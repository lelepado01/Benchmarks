# Benchmarks

Hi Val, the math seems fine, I'm not sure what's wrong... please don't kill meee

The src folder contains the source code for running the experiments, while data analysis is what I did Saturday...

The important files are 
- clean.py which removes all columns which are not the average power/time/loss from the imported logs
- to_square.py which creates a GPUs X ModelSize square matrix containing the results
- compare_ratio.py which calculates the mult. and plots to heatmap

Initial logs are files such as SW_power or MAE_power or the other respective metrics, then get turned to SW_power_cleaned, then to SW_power_cleaned_square, and then visualized

Sorry for the mess but I just took the files and uploaded to a new repo
