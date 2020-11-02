import numpy
from overrides import overrides
from typing import List, Dict

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('cxm_predictor')
class CXMPredictor(Predictor):

    def predict_instance(self, instance: Instance) -> JsonDict:
        output = self._model.forward_on_instance(instance)
        output_json = {
            "id": output["metadata"]["id"],
            "dialogue_id": output["metadata"]["dialogue_id"],
            "turn_index": output["metadata"]["turn_index"],
            "label_probs": output['label_probs'],
            "label_pred": output['label_pred']
        }
        return sanitize(output_json)


    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        output_json_list = []
        for i in range(len(outputs)):
            label_probs_batch = outputs[i]['label_probs']
            label_pred_batch = outputs[i]['label_pred']
            output_json_list.append({
                "id": outputs[i]["metadata"]["id"],
                "dialogue_id": outputs[i]["metadata"]["dialogue_id"],
                "turn_index": outputs[i]["metadata"]["turn_index"],
                "label_probs": label_probs_batch,
                "label_pred": label_pred_batch
            })

        return sanitize(output_json_list)

    def round_probs(self, probs):
        rp = [round(p, 3) for p in probs]
        return rp

    @overrides
    def _json_to_instance(self, inst: JsonDict) -> Instance:
        """
        """
        dialogue_id = inst['dialogue_id']
        turn_index = inst['turn_index']
        prev_turns = inst['prev_turns']
        label_scores = inst['label_scores']
        label_gold = label_scores.index(max(label_scores))

        
        metadata = {}
        metadata['dialogue_id'] = dialogue_id
        metadata['turn_index'] = turn_index
        metadata['prev_turns'] = prev_turns

        metadata['id'] = inst['id']

        utterance = inst['utterance'].strip()


        return self._dataset_reader.text_to_instance(dialogue_id,
                                                     turn_index,
                                                     utterance,
                                                     label_scores,
                                                     label_gold,
                                                     metadata)