#!/usr/bin/env bash

mkdir -p jp
mkdir -p en

DDATA='data_all'
mkdir -p $DDATA
cd $DDATA

# DBDC1-jp-dev
wget -O dbdc1-jp-dev.zip https://sites.google.com/site/dialoguebreakdowndetection/dev_data/dev.zip
unzip -q dbdc1-jp-dev.zip -d dbdc1-jp-dev
rm dbdc1-jp-dev.zip
mv dbdc1-jp-dev/dev/*.log.json dbdc1-jp-dev
rm dbdc1-jp-dev/*.txt
rm -rf dbdc1-jp-dev/dev

# DBDC1-jp-eval
wget -O dbdc1-jp-eval.zip https://sites.google.com/site/dialoguebreakdowndetection/dev_data/eval.zip
unzip -q dbdc1-jp-eval.zip -d dbdc1-jp-eval
rm dbdc1-jp-eval.zip
mv dbdc1-jp-eval/eval/eval/*.log.json dbdc1-jp-eval
rm dbdc1-jp-eval/*.txt
rm -rf dbdc1-jp-eval/eval

# DBDC2-jp-dev
wget -O dbdc2-jp-dev.zip https://sites.google.com/site/dialoguebreakdowndetection2/downloads/DBDC2_dev.zip
unzip -q dbdc2-jp-dev.zip -d dbdc2-jp-dev
rm dbdc2-jp-dev.zip
cp dbdc2-jp-dev/DBDC2_dev/*/*.log.json dbdc2-jp-dev

# DBDC2-jp-eval
wget -O dbdc2-jp-eval.zip https://sites.google.com/site/dialoguebreakdowndetection2/downloads/DBDC2_ref.zip
unzip -q dbdc2-jp-eval.zip -d dbdc2-jp-eval
rm dbdc2-jp-eval.zip
cp dbdc2-jp-eval/DBDC2_ref/*/*.log.json dbdc2-jp-eval

# DBDC3
wget -O dbdc3.zip https://dbd-challenge.github.io/dbdc3/data/DBDC3.zip
unzip -q dbdc3.zip -d dbdc3
rm dbdc3.zip

# DBDC3-jp-eval
mkdir -p dbdc3-jp-eval
cp dbdc3/DBDC3/dbdc3_revised/ja/eval/*/*.log.json dbdc3-jp-eval

# DBDC3-en-dev
mkdir -p dbdc3-en-dev
cp dbdc3/DBDC3/dbdc3_revised/en/dev/*/*.log.json dbdc3-en-dev

# DBDC3-en-eval
mkdir -p dbdc3-en-eval
cp dbdc3/DBDC3/dbdc3_revised/en/eval/*/*.log.json dbdc3-en-eval

# DBDC4-dev
wget -O dbdc4-dev.tgz https://www.dropbox.com/s/g6jb16suq07x9v2/DBDC4_dev_20190312.tgz
tar -xzf dbdc4-dev.tgz
rm dbdc4-dev.tgz
mv DBDC4_dev_20190312 dbdc4-dev

# DBDC4-jp-dev
mkdir -p dbdc4-jp-dev
for dir in dbdc4-dev/ja/dbd_livecompe_dev/*; do for f in $dir/*; do ff=${f##*/}; dd=${dir##*/}; cp "$f" "dbdc4-jp-dev/${ff//messages.html/LIVE-DEV-$dd}"; done; done

# DBDC4-en-dev
mkdir -p dbdc4-en-dev
cp dbdc4-dev/en/*.log.json dbdc4-en-dev

# DBDC4-eval
wget -O dbdc4-eval.tgz https://www.dropbox.com/s/5e7aqo1i80tqnxn/DBDC4_eval_20200314.tgz
tar -xzf dbdc4-eval.tgz
rm dbdc4-eval.tgz
mv DBDC4_eval_20200314/ dbdc4-eval

# DBDC4-jp-eval
mkdir -p ../jp/eval_all
mkdir -p dbdc4-jp-eval
for dir in dbdc4-eval/jp/*[A-Z]*; do cp "${dir}"/*.log.json dbdc4-jp-eval; done
for dir in dbdc4-eval/jp/dbd_livecompe_eval/*; do for f in $dir/*; do ff=${f##*/}; dd=${dir##*/}; cp "$f" "dbdc4-jp-eval/${ff//messages.html/LIVE-EVAL-$dd}"; done; done
cp dbdc4-jp-eval/*.log.json ../jp/eval_all

# DBDC4-en-eval
mkdir -p ../en/eval_all
mkdir -p dbdc4-en-eval
cp dbdc4-eval/en/*.log.json dbdc4-en-eval
cp dbdc4-en-eval/*.log.json ../en/eval_all

# JP-train
mkdir -p ../jp/train_all
for d in dbdc1-jp-dev dbdc1-jp-eval dbdc2-jp-dev dbdc2-jp-eval dbdc4-jp-dev dbdc3-en-dev dbdc3-en-eval dbdc4-en-dev; do cp $d/*.log.json ../jp/train_all; done

# JP-dev
mkdir -p ../jp/dev_all
cp dbdc3-jp-eval/*.log.json ../jp/dev_all

# EN-train
mkdir -p ../en/train_all
for d in dbdc3-en-dev dbdc4-en-dev dbdc1-jp-dev dbdc1-jp-eval dbdc2-jp-dev dbdc2-jp-eval dbdc3-jp-eval dbdc4-jp-dev; do cp $d/*.log.json ../en/train_all; done

# EN-dev
mkdir -p ../en/dev_all
cp dbdc3-en-eval/*.log.json ../en/dev_all

# go back to data directory
cd ..

# generate json and jsonl files
for m in train dev eval; do python data_conversion.py --lang jp --mode $m; python data_conversion.py --lang en --mode $m; done

mv jp/train.json jp/train_d.json
mv en/train.json en/train_d.json

# go to main directory
cd ..
mkdir -p evaluation/eval_script
wget -O evaluation/eval_script/eval.py https://raw.githubusercontent.com/dbd-challenge/dbdc4/master/prog/eval/eval.py

# Done
echo "data downloading & preprocessing: done"
