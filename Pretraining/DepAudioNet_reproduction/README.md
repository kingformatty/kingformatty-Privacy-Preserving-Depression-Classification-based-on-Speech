**Credit**

This repository relates to our work in the EUSIPCO 2021 paper, "Gender Bias in Depression Detection Using Audio Features", https://arxiv.org/abs/2010.15120

**Prerequisites**

This was develpoed for Ubuntu (18.04) with Python 3

Install miniconda and load the environment file from environment.yml file

`conda env create -f environment.yml`

(Or should you prefer, use the text file)

Activate the new environment: `conda activate myenv`

**Experiment Setup**

Use the config_pretrain.py file to set experiment preferences and locations of the code, workspace, and dataset directories. 

Call `./run_pretrain.sh` for training

Configuration File Details:
- `FEATURE_exp` - Determines feature type, and our exp is based on mel-spectrogram
- `model` - Default None, CPC/wav2vec are also supported
- `CLASS_WEIGHT` - Weight loss for ND and D class according to number of sample
- `SUB_SAMPLE_ND_CLASS` - Make len(D) == len(ND) in a batch
- `CROP` - CROP each utterance to the shortest length in each subset (MD, FD, MND, FND)
- `SPLIT_BY_GENDER` - Not used
- `FEATURE_DIMENSIONS` - Segment Length
- `FREQ_BINS` - Feature Dimension
- `BATCH_SIZE` - batch size
- `SVN` - Normalization
- `LEARNING_RATE`
- `learn_rate_factor`
- `SEED`
- `TOTAL_EPOCHS`
- `TOTAL_ITERATIONS`
- `ITERATIOn_EPOCH`
- `SUB_DIR` - experiment directory
- `EXP_RUNTHROUGH` - number of model for training
- `hidden_size`
- etc.(not applicabale)
- `threshold` - fscore/total, metric to determine best model
- `MIN_CROP` - Crop all utt into shortest length of the dataset
- `Pretrain` - whether resume pretrained model
- `Resume_model_from` - Resumed model path

Additional Argument:
- `temperature` - scaling factor in pretraining for vector multiplication
- `BN_per_epoch_train` - Batch num for pretrain
- `BN_dev` - Batch num for Validation
- `random_speaker` - samples in batch to be distinct(false)/random(true)
- `Augmentation` - whether use external signal level augmentation dataset (extracted)
- `single_augmentation` - use all augmentation or only one
- Augmentation techniques configurations -> omitted
- `AUG_DATA_PATH` - External signal level augmentation dataset path (extracted)

