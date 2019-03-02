# ece271b-project

## Files
[DRLoader.py](DRLoader.py) - Video Dataset loader for Pytorch. Handles reading video frame by frame and creating batches.

[download_split_dataset.py](download_split_dataset.py) - Downloads and splits UCF11 dataset into proper directory structure.

[model.py](model.py) - Pytorch models for dLSTM, dVGG, and VGG. 

[net.py](net.py) - Runs training/evaulation for dLSTM/dVGG network. 

[net_reg.py](net_reg.py) - Runs training/evaluation for LSTM/VGG network. 

## To run
```
python download_split_dataset.py
python net.py --batchSize 2 --windowSize 15 --h_dim 64
python net_reg.py --batchSize 2 --windowSize 15 --h_dim 64
```

### To avoid out of memory errors
```
python net.py --batchSize 2 --windowSize 15 --h_dim 64
```
