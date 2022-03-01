# Privacy-Preserving-Depression-Classification-based-on-Speech

This repo is the course project for ECE209: Secure and Trustworthy Edge Computing Systems
Team Member: Jinhan Wang, Tianyi Yang, Zichun Chai

## Pre-requisites
This work is based on Andrew Bailey's DepAudioNet Reproduction and pre-processing repo.   
https://github.com/kingformatty/DepAudioNet_reproduction.git  
https://github.com/adbailey1/daic_woz_process.git  


Please setup your environment first following the guideline.

## Dataset
For this experiments, DAIC-WOZ dataset is used which can be otrained through The University of Southern California (http://dcapswoz.ict.usc.edu) by signing a license agreement. The dataset is roughly 135GB. There are several errors when collecting and preparing the dataset, so please use the provided repo for pre-processing. 

For pretraining, Librispeech (360 hours) is used, this corpus can be obtained through openslr https://www.openslr.org/12/  

## Folder Structure

### [Classification/DepAudioNet_reproduction](Classification/DepAudioNet_reproduction)  

Baseline and Experiments with resumed pretrained model can be run under this repository.  



### [Pretraining/DepAudioNet_reproduction](Pretraining/DepAudioNet_reproduction)  
Pretraining pipeline without speaker disentanglement. Currently, time-masking/frequency-masking/Specaug/vtlp/pitch-perturbation/noise-perturbation/volume-perturbation are supported.

### [Pretraining/DepAudioNet_reproduction_dise](Pretraining/DepAudioNet_reproduction_disentangle)  
Pretraining pipeline with speaker disentanglement. Supported Augmentation techniques are the same as above.

### [SPK](SPK)
Speaker related experiment directory

### [librispeech preparation](librispeech_prepration_scripts)
Librispeech preparation scripts for pretraining usecase.

### [data preparation/feature extraction](depression_prep)
Feature extraction pipeline with augmentation options(signal level, feature level doesn't need to be extracted)
