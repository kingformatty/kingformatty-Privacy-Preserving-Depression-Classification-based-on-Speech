#! /usr/bin/env bash


cat flac.scp | while read line
  do
    #echo $line
    new_name=$(echo $line  | sed "s:\(.*\)\(train-clean-360\)\(.*\)\(flac\):\1train-clean-360_wav\3\wav:")
    echo $new_name
    echo /$line    
    ffmpeg -i /$line /$new_name
    #break
  done
