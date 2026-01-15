# DL-1D-CNN
This repository contains the official implementation of our paper, "VoIP Call Identification via a Dual-Level 1D-CNN with Frame and Utterance Features".

## Requirements
Installing dependencies:
```
pip install -r requirements.txt
```

## Run the training procedure
Run the `train.py` to train the model on the VPCID dataset:
```
python3.8 train.py -o ./results/save_name --gpu 0 --batch_size 128 --num_epochs 100 --case case0
```
- The `--case` option supports `case0`–`case5`, `case6-1`, and `case6-2`.

## Citation
If you find this work useful in your research, please consider citing this paper.
