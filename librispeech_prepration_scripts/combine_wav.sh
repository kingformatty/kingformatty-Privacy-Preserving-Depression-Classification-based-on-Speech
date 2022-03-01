#! /usr/bin/env bash

folder_list=`ls ../train-clean-360_wav`

for file in $folder_list
do
  echo $file
  cd ../train-clean-360_wav/$file
  audio_files=$(find -iname '*.wav')
  #combine all files of an audio together and save it into correct directory
  output_dir=../../train-clean-360_wav_combined_mod/${file}_P
  if [ ! -d $output_dir ]; then
    mkdir -p $output_dir 
  fi
  sox $(for f in $audio_files; do echo "$f"; done) $output_dir/${file}_AUDIO.wav
  cd ../../scripts 
done
