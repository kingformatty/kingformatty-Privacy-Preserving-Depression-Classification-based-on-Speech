import os

# Use this string to write a brief detail about the current experiment. This
# string will be saved in a logger for this particular experiment
EXPERIMENT_BRIEF = ''
# Set to complete to use all the data
# Set to sub to use training/dev sets only
# Network options: custom or custom_att (to use the attention mechanism)
# Set to complete to use all the data
# Set to sub to use training/dev sets only
# Network options: custom or custom_att (to use the attention mechanism)
EXPERIMENT_DETAILS = {'FEATURE_EXP': 'mel',
                      'CLASS_WEIGHTS': False,
                      'USE_GENDER_WEIGHTS': False,
                      'SUB_SAMPLE_ND_CLASS': False,  # Make len(dep) == len(
                      # ndep)
                      'CROP': False,
                      'OVERSAMPLE': False,
                      'SPLIT_BY_GENDER': False,#True,  # Only for use in test mode
                      'FEATURE_DIMENSIONS': 120,
                      'FREQ_BINS': 40,
                      'BATCH_SIZE': 128,
                      'SVN': True,
                      'LEARNING_RATE': 1e-3,
                      'SEED': 1000,
                      'TOTAL_EPOCHS': 50,
                      'TOTAL_ITERATIONS': 3280,
                      'ITERATION_EPOCH': 1,
                      'SUB_DIR': 'exp_speaker_dise/TM_tmp10_distinct_MI_3e-4_3_lam_1e2_BN2000_BS128',
                      'temperature': 10,
                      'EXP_RUNTHROUGH': 1,
                      'BN_per_epoch_train': 2000,
                      'BN_dev': 500,
                      'SAVE_MODEL_FROM':1,
                      'random_speaker': False,
                      'Augmentation': False,
                      'hidden_size': 192,
                      'bidirectional': False,
                      'stat_pool': False,
                      'stat_pool_var':False,
                      'normalization' : False }#augmentation strategies: 'noise_10dB','noise_20dB','noise_10_20dB'
# Determine the level of crop, min file found in training set or maximum file
# per set (ND / D) or (FND, MND, FD, MD)
MIN_CROP = False
# Determine whether the experiment is run in terms of 'epoch' or 'iteration'
ANALYSIS_MODE = 'epoch'


#spk config
lr_MI=3e-4
iter_MI=3
lambda_MI=1e2
#pretraining config
Pretrain = True
Resume_model_from = '/home/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet/mel_svn_exp/exp_BN_20/TM_tmp10_distinct/model/1/md_89_epochs.pth'
data_saver = False
batch_norm = False
freeze = False
# How to calculate the weights: 'macro' uses the number of individual
# interviews in the training set (e.g. 31 dep / 76 non-dep), 'micro' uses the
# minimum number of segments of both classes (e.g. min_num_seg_dep=35,
# therefore every interview in depressed class will be normalised according
# to 35), 'both' combines the macro and micro via the product, 'instance'

# uses the total number of segments for each class to determine the weights (
# e.g. there could be 558 dep segs and 440 non-dep segs).
WEIGHT_TYPE = 'instance'

# Set to 'm' or 'f' to split into male or female respectively
# Otherwise set to '-' to keep both genders in the database
GENDER = '-'

# There are configurations for augmentation
#TM
use_time_mask = True
time_mask_size = 20
time_mask_random = True #not applicable for now
use_spec_aug = False #detemine whether to use spec_aug
#time warp
use_time_warp = False
max_time_warp = 5 
inplace = False
resize_mode = "PIL"
#freq mask
max_freq_width = 0
n_freq_mask = 0
replace_with_zero = False 
mode = "random"
#time mask
max_time_width = 5 
n_time_mask = 2




# These values should be the same as those used to create the database
# If raw audio is used, you might want to set these to the conv kernel and
# stride values
WINDOW_SIZE = 1024
HOP_SIZE = 512
OVERLAP = int((HOP_SIZE / WINDOW_SIZE) * 100)

FEATURE_FOLDERS = ['audio_data', 'logmel']
EXP_FOLDERS = ['log', 'model', 'condor_logs']

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'text':
    FEATURE_FOLDERS = None
else:
    FEATURE_FOLDERS = ['audio_data', 'logmel']
EXP_FOLDERS = ['log', 'model', 'condor_logs']

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'logmel' or EXPERIMENT_DETAILS[
    'FEATURE_EXP'] == 'MFCC' or EXPERIMENT_DETAILS['FEATURE_EXP'] == \
        'MFCC_concat':
    if EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
        FOLDER_NAME = f"BKGND_{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_{str(EXPERIMENT_DETAILS['FREQ_BINS'])}"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}_{str(EXPERIMENT_DETAILS['FREQ_BINS'])}"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and not \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_" \
                      f"{str(EXPERIMENT_DETAILS['FREQ_BINS'])}_with_backgnd"
else:
    FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}"

if EXPERIMENT_DETAILS['SVN']:
    FOLDER_NAME = FOLDER_NAME + '_svn_exp'
else:
    FOLDER_NAME = FOLDER_NAME + '_exp'

if EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']:
    EXPERIMENT_DETAILS['SUB_DIR'] = EXPERIMENT_DETAILS['SUB_DIR'] + '_gen'


DATASET ='/home/jinhan/Dataset/librispeech/LibriSpeech360/train-clean-100_wav_combined_mod'
WORKSPACE_MAIN_DIR ='/home/../data/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet'
SPK_EMB_DIR=WORKSPACE_MAIN_DIR+'/mel_svn_exp/embedding/'
#NOISE_10dB_path='/home/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet_noise_10dB/mel_svn_exp'
#NOISE_20dB_path='/home/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet_noise_20dB/mel_svn_exp'
AUG_DATA_path='/home/../data/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet_volume/mel_svn_exp'

WORKSPACE_FILES_DIR = '/home/jinhan/DepAudioNet_reproduction_disentangle'
TRAIN_SPLIT_PATH ='/home/jinhan/Dataset/librispeech/LibriSpeech360/labels/train_split_Depression_AVEC2017.csv'
DEV_SPLIT_PATH ='/home/jinhan/Dataset/librispeech/LibriSpeech360/labels/dev_split_Depression_AVEC2017.csv' 
TEST_SPLIT_PATH ='/home/jinhan/Dataset/librispeech/LibriSpeech360/labels/full_test_split.csv'
FULL_TRAIN_SPLIT_PATH ='/home/jinhan/Dataset/librispeech/LibriSpeech360/labels/full_train_split_Depression_AVEC2017.csv'
COMP_DATASET_PATH ='/home/jinhan/Dataset/librispeech/LibriSpeech360/labels/complete_Depression_AVEC2017.csv'
"""

DATASET ='/home/jinhan/Dataset/daic-woz-old/data'
WORKSPACE_MAIN_DIR ='/home/jinhan/Dataset/daic-woz-old/audio_feats/feats_DepAudioNet'
#NOISE_10dB_path='/home/jinhan/Dataset/daic-woz-old/audio_feats/feats_DepAudioNet_noise_10dB/mel_snv_exp'
#NOISE_20dB_path='/home/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet_noise_20dB/mel_svn_exp'
WORKSPACE_FILES_DIR = '/home/jinhan/DepAudioNet_reproduction'
TRAIN_SPLIT_PATH ='/home/jinhan/Dataset/daic-woz-old/labels/train_split_Depression_AVEC2017.csv'
DEV_SPLIT_PATH ='/home/jinhan/Dataset/daic-woz-old/labels/dev_split_Depression_AVEC2017.csv' 
TEST_SPLIT_PATH ='/home/jinhan/Dataset/daic-woz-old/labels/full_test_split.csv'
FULL_TRAIN_SPLIT_PATH ='/home/jinhan/Dataset/daic-woz-old/labels/full_train_split_Depression_AVEC2017.csv'
COMP_DATASET_PATH ='/home/jinhan/Dataset/daic-woz-old/labels/complete_Depression_AVEC2017.csv'
"""


"""
DATASET ='/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb/CN_combined_mod'
WORKSPACE_MAIN_DIR ='/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb/audio_feats/feats_DepAudioNet'
NOISE_10dB_path='/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb//audio_feats/feats_DepAudioNet_noise_10dB/mel_svn_exp'
#NOISE_20dB_path='/home/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet_noise_20dB/mel_svn_exp'
WORKSPACE_FILES_DIR = '/home/jinhan/DepAudioNet_reproduction'
TRAIN_SPLIT_PATH ='/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb/labels/train_split_Depression_AVEC2017.csv'
DEV_SPLIT_PATH ='/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb/labels/dev_split_Depression_AVEC2017.csv' 
TEST_SPLIT_PATH ='/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb/labels/full_test_split.csv'
FULL_TRAIN_SPLIT_PATH ='/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb//labels/full_train_split_Depression_AVEC2017.csv'
COMP_DATASET_PATH ='/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb//labels/complete_Depression_AVEC2017.csv'
"""



LOG_DIR = WORKSPACE_MAIN_DIR+'/'+FOLDER_NAME+'/'+EXPERIMENT_DETAILS['SUB_DIR']+'/log'
