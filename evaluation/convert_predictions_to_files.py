#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import glob
import math
import os

label2idx = {'O': 0, 'T': 1, 'X': 2}
idx2label = ['O', 'T', 'X']

def load_jsonl_data(data_dir):
	data = []
	with open(data_dir, 'r', encoding='utf-8') as lf:
		for line in lf:
			data.append(json.loads(line))

	lf.close()
	return data

def create_labels_dir(dir_to_create):
	if not os.path.exists(dir_to_create):
		os.makedirs(dir_to_create)

def write_json_to_dir(data, w_dir):
	wf = open(w_dir, 'w')
	json.dump(data, wf, indent=2, ensure_ascii=False)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--eval_file", type=str, default='eval.jsonl',
						help='eval set')

	args = parser.parse_args()
	realpath_dir = os.path.dirname(os.path.realpath(__file__))
	model_parent_path = os.path.abspath(os.path.join(args.eval_file, '..'))
	model_name = model_parent_path[model_parent_path.rfind('/')+1:]

	save_dir = realpath_dir + '/pred_label_files/labels_' + model_name

	eval_data = load_jsonl_data(args.eval_file)

	all_dialogues = {}

	for inst in eval_data:
		dialogue_id = inst['dialogue_id']
		turn_index = inst['turn_index']
		label_probs = inst['label_probs']
		label_pred_index = label_probs.index(max(label_probs))

		dinst = {'turn-index': turn_index,
				 'labels': [{
				 	'breakdown': idx2label[label_pred_index],
				    'prob-O' : label_probs[0],
				    'prob-T' : label_probs[1],
				    'prob-X' : label_probs[2]
				 }]
		}

		if dialogue_id in all_dialogues:
			all_dialogues[dialogue_id] += [dinst]
		else:
			all_dialogues[dialogue_id] = [dinst]

	create_labels_dir(save_dir)

	for dialogue in all_dialogues:
		jinst = {
			'dialogue-id': dialogue,
			'turns': all_dialogues[dialogue]
		}
		save_to = save_dir + '/' + dialogue + '.labels.json'
		write_json_to_dir(jinst, save_to)

if __name__ == '__main__':
    main()