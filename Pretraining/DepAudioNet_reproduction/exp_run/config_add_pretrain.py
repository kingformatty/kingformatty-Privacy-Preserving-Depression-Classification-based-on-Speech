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
                      'weight_tensor': [1,1],
                      'USE_GENDER_WEIGHTS': False,
                      'Gender_min': False,
                      'SUB_SAMPLE_ND_CLASS': False,  # Make len(dep) == len(
                      # ndep)
                      'CROP': False,
                      'OVERSAMPLE': False,
                      'SPLIT_BY_GENDER': False,#True,  # Only for use in test mode
                      'custommel': False,
                      'FEATURE_DIMENSIONS': 120,
                      'FREQ_BINS': 40,
                      'BATCH_SIZE': 256,
                      'SVN': True,
                      'LEARNING_RATE': 1e-2,
                      'SEED': 1000,
                      'TOTAL_EPOCHS': 100,
                      'TOTAL_ITERATIONS': 3280,
                      'ITERATION_EPOCH': 1,
                      'SUB_DIR': '/home/jinhan/Dataset/converge_merged_audio_feats/feats_DepAudioNet/mel_svn_exp/No_pretrain_norm',
                      'EXP_RUNTHROUGH':1, 
                      'hidden_size': 128,
                      'bidirectional': False,
                      'stat_pool': False,
                      'stat_pool_var':False,
                      'normalization' : True ,  
                      'weight_decay': 0
                      }
threshold = 'fscore' # fscore or total

dataset = 'orig' # 'aug' or 'orig'
#test epoch

#test_epoch = '_5_epochs'
#test mode
#test_mode = 'ave'
#test_mode corresponding config
#out_name = 'md_100_avg.pth'
#num = 5
#last_epoch = 99

# Determine the level of crop, min file found in training set or maximum file
# per set (ND / D) or (FND, MND, FD, MD)
MIN_CROP = False
# Determine whether the experiment is run in terms of 'epoch' or 'iteration'
ANALYSIS_MODE = 'epoch'

#pretraining configurations
Pretrain = False
#Resume_model_from = '/home/jinhan/Dataset/cn-celeb/cn-celeb/CN-Celeb/audio_feats/feats_DepAudioNet/mel_svn_exp/exp/noise_tmp0.1_rand_norm/model/1/md_94_epochs.pth'
Resume_model_from = '/home/jinhan/Dataset/librispeech/Librispeech360/audio_feats/feats_DepAudioNet/mel_svn_exp/exp_BN_20/Specaug_tmp0.1_random/model/1/md_92_epochs.pth'
freeze = False
data_saver =  False
batch_norm = False
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

if EXPERIMENT_DETAILS['SVN']:
    FOLDER_NAME = FOLDER_NAME + '_svn_exp'
else:
    FOLDER_NAME = FOLDER_NAME + '_exp'

if EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']:
    EXPERIMENT_DETAILS['SUB_DIR'] = EXPERIMENT_DETAILS['SUB_DIR'] + '_gen'

if dataset == 'aug':
    data_path = 'data_aug'
    workspace_path = 'feats_DepAudioNet_noise_20dB_dep'
    label_path = 'labels_aug'
elif dataset == 'orig':
    data_path = 'data'
    workspace_path = 'feats_DepAudioNet'
    label_path = 'all'
 

DATASET ='/home/jinhan/Dataset/converge_merged_audio'+data_path
WORKSPACE_MAIN_DIR ='/home/jinhan/Dataset/converge_merged_audio_feats/'+ workspace_path
WORKSPACE_FILES_DIR = '/home/jinhan/DepAudioNet_reproduction'
TRAIN_SPLIT_PATH ='/home/jinhan/Dataset/converge/depaudio_labels/'+label_path+'/train_converge.csv'
DEV_SPLIT_PATH ='/home/jinhan/Dataset/converge/depaudio_labels/'+label_path+'/dev_converge.csv' 
TEST_SPLIT_PATH ='/home/jinhan/Dataset/converge/depaudio_labels/'+label_path+'/test_converge.csv'
FULL_TRAIN_SPLIT_PATH ='/home/vijaysumaravi/Documents/database/daic-woz-old/'+label_path+'/full_train_split_Depression_AVEC2017.csv'
COMP_DATASET_PATH ='/home/vijaysumaravi/Documents/database/daic-woz-old/'+label_path+'/complete_Depression_AVEC2017.csv'

