# CXM  #

This repository contains the source code of the 
paper [A Co-Attentive Cross-Lingual Neural Model for Dialogue Breakdown Detection](https://www.aclweb.org/anthology/2020.coling-main.371.pdf).

### Publication ###
If you use the source code or models from this work, please cite our paper:
```
@inproceedings{lin-etal-2020-cxm,
  author    = "Lin, Qian and Kundu, Souvik and Ng, Hwee Tou",
  title     = "A Co-Attentive Cross-Lingual Neural Model for Dialogue Breakdown Detection",
  booktitle = "Proceedings of COLING",
  year      = "2020",
}
```


### Requirements ###

Install the packages listed in the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

Install allennlp
```bash
bash install_allennlp.sh
```

### Data ###

Refer to `data/README.md` for instructions of data downloading and preprocessing.

The processed data files will be located at `data/en` and `data/jp` for English track and Japanese track, respectively.


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

We provide trained models. They can be downloaded by running `bash download_trained_models.sh`.

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
The evaluation script will be downloaded during the process of data downloading and preprocessing.

First, convert prediction file to seperate json files:

```bash
cd evaluation
model_dir = "en_cxm_d"
python convert_predictions_to_files.py --eval_file ../models/$model_dir/eval_pred.jsonl
```
Then run the evaluation script:
```bash
python2 eval_script/eval.py -t 0.0 -p ../data/en/eval_all/ -o pred_label_files/labels_$model_dir
```



### License ###

The code and models in this repository are licensed under the GNU General Public License Version 3. For commercial use of this code and models, separate commercial licensing is also available. Please contact:
* Qian Lin ([qlin@u.nus.edu](mailto:qlin@u.nus.edu))
* Souvik Kundu ([souvik@u.nus.edu](mailto:souvik@u.nus.edu))
* Hwee Tou Ng ([nght@comp.nus.edu.sg](mailto:nght@comp.nus.edu.sg))
