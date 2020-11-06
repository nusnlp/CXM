#!/usr/bin/env bash

wget -O allennlp.tar.gz https://files.pythonhosted.org/packages/76/48/2e5226b10b1f7894c057af75b7b7d5f9db433d10ecea99d14a1bc6277f80/allennlp-0.9.1.dev20200220.tar.gz
tar -xzf allennlp.tar.gz
rm allennlp.tar.gz
mv allennlp-0.9.1.dev20200220 allennlp
cd allennlp
sed -i '208s/)/, encoding="utf-8")/' allennlp/commands/predict.py
sed -i '105s/None,/None, return_token_type_ids=True,/' allennlp/data/tokenizers/pretrained_transformer_tokenizer.py
pip install --editable . 
cd ..
rm -rf allennlp
