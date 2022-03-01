import torchaudio
from speechbrain.pretrained import EncoderClassifier
import numpy as np
import pickle
import os
import pdb

audio_path= '/data/jinhan/Dataset/daic-woz-old/data'
embedding_dir = '/data/jinhan/Dataset/daic-woz-old/audio_feats/feats_DepAudioNet/mel_snv_exp/embedding/'
audio_list = []
for root, dirs, files in os.walk(audio_path):
    for audio_folder in dirs:
        import pdb
        folder_path = os.path.join(root,audio_folder)
        audio_file = os.listdir(folder_path) 
        for sing_file in audio_file:
                if sing_file.find('AUDIO.wav')!=-1:
                    audio_list.append(os.path.join(folder_path,sing_file))
audio_list = audio_list[:189]
#x_vector extraction
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
for audio_path in audio_list:
    signal, fs =torchaudio.load(audio_path)
    embeddings = classifier.encode_batch(signal)
    import pdb
    elements = audio_path.split('/')[-1]
    spk_id = elements.split('_')[0]
    output_filename = spk_id+'_emb.pickle'
    emb_out_file = os.path.join(embedding_dir,output_filename)
    with open(emb_out_file,'wb') as file:
        pickle.dump(embeddings,file)
    print('Finish {}'.format(output_filename)) 
