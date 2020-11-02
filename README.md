# CXM  #

This repository contains the source code of the 
paper A Co-Attentive Cross-Lingual Neural Model for Dialogue Breakdown Detection.


### Requirements ###

Install the packages listed in the `requirements.txt` file.
```bash
pip install - r requirements.txt
```


### Data ###

Instructions of downloading and preprocessing data will be available soon.

The processed data files are moved to `data/en` and `data/jp` for English track and Japanese track, respectively.

The processed training sets `data/en/train_d.json` and `data/jp/train_d.json` follow the CXM-D setting, which include data in both languages.

The DBDC4 labeled evaluation data files are moved to `data/english/eval_all` and `data/japanese/eval_all` for English track and Japanese track, respectively.


### Training ###

We provide training configuration files in `training_configs`. Modify the paths to data inside the configuration files.

For English track:
```bash
allennlp train -s models/en_cxm_d --include-package cxm training_configs/en_cxm_d.json
```

Similarly for Japanese track:
```bash
allennlp train -s models/jp_cxm_d --include-package cxm training_configs/jp_cxm_d.json
```


### Prediction ###

```bash
model_dir = "en_cxm_d"
allennlp predict models/$model_dir/model.tar.gz data/en/eval.jsonl \
                    --output-file models/$model_dir/eval_pred.jsonl \
                    --batch-size 2 \
                    --cuda-device 0 \
                    --predictor cxm_predictor \
                    --include-package cxm \
                    --silent
```


### Evaluation ###

For evaluation, we follow the use of official DBDC evaluation script `evaluation/eval_script/eval.py`.

First, convert prediction file to seperate json files:

```bash
cd evaluation
model_dir = "en_cxm_d"
python convert_predictions_to_files.py --eval_file ../models/$model_dir/eval_pred.jsonl
```
Then run the evaluation script:
```bash
python2 eval_script/eval.py -t 0.0 -p ../data/english/eval_all/ -o pred_label_files/labels_$model_dir
```



### License ###

The code and models in this repository are licensed under the GNU General Public License Version 3. For commercial use of this code and models, separate commercial licensing is also available. Please contact:
* Qian Lin ([qlin@u.nus.edu](mailto:qlin@u.nus.edu))
* Souvik Kundu ([souvik@u.nus.edu](mailto:souvik@u.nus.edu))
* Hwee Tou Ng ([nght@comp.nus.edu.sg](mailto:nght@comp.nus.edu.sg))
