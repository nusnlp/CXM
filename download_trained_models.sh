#!/usr/bin/env bash

MODEL_DIR="trained_models"
mkdir -p $MODEL_DIR

cd $MODEL_DIR
mkdir -p en
mkdir -p jp
cd en
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILE_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rsjV2DiF0gePgoCejOUWSXGaGKByTECU" -O model.tar.gz && rm -rf /tmp/cookies.txt

cd ..
cd jp
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILE_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_j41rMbb7FybxfHKW1Ae0dgf77Lhb39f" -O model.tar.gz && rm -rf /tmp/cookies.txt

cd ../..