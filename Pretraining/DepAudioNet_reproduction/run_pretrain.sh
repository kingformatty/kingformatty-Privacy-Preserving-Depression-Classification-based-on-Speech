#! /usr/bin/env bash

model_config=


CUDA_VISIBLE_DEVICES=2 python3 IDL_pretrain.py train --validate --vis --cuda --debug --position=2


