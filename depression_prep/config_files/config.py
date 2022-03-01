import os
import numpy as np

# FEATURE_EXP: logmel, mel, raw, MFCC, MFCC_concat, or text
# WHOLE_TRAIN: This setting is for mitigating the variable length of the data
# by zero padding
# SVN will normalise every file to mean=0 and standard deviation=1
EXPERIMENT_DETAILS = {'FEATURE_EXP': 'mel',
                      'FREQ_BINS': 40,
                      'DATASET_IS_BACKGROUND': False,
                      'WHOLE_TRAIN': False,
                      'WINDOW_SIZE': 1024,
                      'OVERLAP': 50,
                      'SVN': True,
                      'SAMPLE_RATE': 16000,
                      'REMOVE_BACKGROUND': True,
                      'Augmentation': 'pitch'} #speed, vtlp, pitch, noise, volume, noise_volume
                      #'Noise_aug_SNR': 10}
#TODO VTLP
# Set True to split data into genders
GENDER = False
WINDOW_FUNC = np.hanning(EXPERIMENT_DETAILS['WINDOW_SIZE'])
FMIN = 0
FMAX = EXPERIMENT_DETAILS['SAMPLE_RATE'] / 2
HOP_SIZE = EXPERIMENT_DETAILS['WINDOW_SIZE'] -\
           round(EXPERIMENT_DETAILS['WINDOW_SIZE'] * (EXPERIMENT_DETAILS['OVERLAP'] / 100))

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'text':
    FEATURE_FOLDERS = None
else:
    FEATURE_FOLDERS = ['audio_data', EXPERIMENT_DETAILS['FEATURE_EXP']]


'''
DATASET = '/home/jinhan/Dataset/librispeech/LibriSpeech360/train-clean-360_wav_combined_mod'
#WORKSPACE_MAIN_DIR = '/home/../../data/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet_'+EXPERIMENT_DETAILS['Augmentation']
WORKSPACE_MAIN_DIR='/home/../../data/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet_pitch'
WORKSPACE_FILES_DIR = '/home/jinhan/depression_prep/config_files'
TRAIN_SPLIT_PATH ='/home/jinhan/Dataset/librispeech/LibriSpeech360/labels/train_split_Depression_AVEC2017.csv' 
DEV_SPLIT_PATH = '/home/jinhan/Dataset/librispeech/LibriSpeech360/labels/dev_split_Depression_AVEC2017.csv'
TEST_SPLIT_PATH = '/home/jinhan/Dataset/librispeech/LibriSpeech360/labels/full_test_split.csv'
FULL_TRAIN_SPLIT_PATH = '/home/jinhan/Dataset/librispeech/LibriSpeech360/labels/full_train_split_Depression_AVED2017.csv'
COMP_DATASET_PATH = '/home/jinhan/Dataset/librispeech/LibriSpeech360/labels/complete_Depression_AVEC2016.csv'
'''


DATASET = '/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb/CN_combined_mod'
WORKSPACE_MAIN_DIR = '/home/../../data/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb/audio_feats/feats_DepAudioNet_'+EXPERIMENT_DETAILS['Augmentation']
WORKSPACE_FILES_DIR = '/home/jinhan/depression_prep/config_files'
TRAIN_SPLIT_PATH ='/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb/labels/train_split_Depression_AVEC2017.csv' 
DEV_SPLIT_PATH = '/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb/labels/dev_split_Depression_AVEC2017.csv'
TEST_SPLIT_PATH = '/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb/labels/full_test_split.csv'
FULL_TRAIN_SPLIT_PATH = '/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb/labels/full_train_split_Depression_AVED2017.csv'
COMP_DATASET_PATH = '/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb/labels/complete_Depression_AVEC2016.csv'


