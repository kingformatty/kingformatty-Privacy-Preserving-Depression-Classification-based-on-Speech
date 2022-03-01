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
                      'model': None,
                      'CLASS_WEIGHTS': False,
                      'USE_GENDER_WEIGHTS': False,
                      'SUB_SAMPLE_ND_CLASS': True,  # Make len(dep) == len(
                      # ndep)
                      'CROP': True,
                      'OVERSAMPLE': False,
                      'SPLIT_BY_GENDER': False,  # Only for use in test mode
                      'FEATURE_DIMENSIONS': 120,
                      'FREQ_BINS': 40,
                      'BATCH_SIZE': 20,
                      'SNV': True,
                      'LEARNING_RATE': 1e-4,
                      'learn_rate_factor': 2,
                      'SEED': 1000,
                      'TOTAL_EPOCHS': 100,
                      'TOTAL_ITERATIONS': 3280,
                      'ITERATION_EPOCH': 1,
                      'SUB_DIR': 'final/IDL/vtlp_tmp10_rand_no_sort_100_1e-4_BN/',
                      'EXP_RUNTHROUGH': 5,
                      'hidden_size': 128,
                      'bidirectional': False,
                      'stat_pool': False,
                      'stat_pool_var': False,
                      'normalization': False}

threshold = 'fscore'
# Determine the level of crop, min file found in training set or maximum file
# per set (ND / D) or (FND, MND, FD, MD)
MIN_CROP = True
# Determine whether the experiment is run in terms of 'epoch' or 'iteration'
ANALYSIS_MODE = 'epoch'

#Pretraining Model Configuration
Pretrain = True
Resume_model_from = '/home/../../data/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet/mel_svn_exp/final/IDL/vtlp_tmp10_random_no_sort/model/1/md_100_epochs.pth'
#Resume_model_from = '/home/../../data/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet/mel_svn_exp/CPC/ts5_seq40/model/1/md_53_epochs.pth'
#Resume_model_from = '/data/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet/mel_svn_exp/kmeans/old/TM_dist_89_cluster_20_1e-3_BN_4300_epoch100_pre/model/1/md_100_epochs.pth'
#Resume_model_from = '/data/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet/mel_svn_exp/SLR/TM_tmp0.1_norm_distinct/model/1/md_96_epochs.pth'
#Resume_model_from = '/home/../../data/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet/mel_svn_exp/exp/new_noise_tmp10_random/model/1/md_97_epochs.pth'
freeze = False
data_saver = False
batch_norm = True
output_layer_lr = None

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

if EXPERIMENT_DETAILS['SNV']:
    FOLDER_NAME = FOLDER_NAME + '_snv_exp'
else:
    FOLDER_NAME = FOLDER_NAME + '_exp'

if EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']:
    EXPERIMENT_DETAILS['SUB_DIR'] = EXPERIMENT_DETAILS['SUB_DIR'] + '_gen'

DATASET = '/home/../../data/jinhan/Dataset/daic-woz-old/data'
WORKSPACE_MAIN_DIR = '/home/../../data/jinhan/Dataset/daic-woz-old/audio_feats/feats_DepAudioNet'
WORKSPACE_FILES_DIR = '/home/jinhan/workdir/DepAudioNet_reproduction'
TRAIN_SPLIT_PATH = os.path.join(DATASET, 'train_split_Depression_AVEC2017.csv')
DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
# TEST_SPLIT_PATH = os.path.join(DATASET, 'test_split_Depression_AVEC2017.csv')
TEST_SPLIT_PATH = os.path.join(DATASET, 'full_test_split.csv')
FULL_TRAIN_SPLIT_PATH = os.path.join(DATASET, 'full_train_split_Depression_AVEC2017.csv')
COMP_DATASET_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')