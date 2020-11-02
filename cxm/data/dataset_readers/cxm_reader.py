import json
import logging
import numpy as np
from typing import Any, Dict, List, Tuple

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp.data.fields import Field, TextField, LabelField, ArrayField, MetadataField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("cxm_reader")
class DBDCDatasetReader(DatasetReader):
    """
    l2af simple reader
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers

    @overrides
    def _read(self, file_path: str):
        logger.info("Reading file at %s", file_path)
        with open(file_path, encoding='utf-8') as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        for inst in dataset:
            dialogue_id = inst['dialogue_id']
            turn_index = inst['turn_index']
            prev_turns = inst['prev_turns']
            label_scores = inst['label_scores']
            label_gold = label_scores.index(max(label_scores))

            
            metadata = {}
            metadata['dialogue_id'] = dialogue_id
            metadata['turn_index'] = turn_index
            metadata['prev_turns'] = prev_turns

            utterance = inst['utterance'].strip()

            instance = self.text_to_instance(dialogue_id,
                                             turn_index,
                                             utterance,
                                             label_scores,
                                             label_gold,
                                             metadata)
            yield instance


    @overrides
    def text_to_instance(self,
                         dialogue_id: str,
                         turn_index: int,
                         utterance: str,
                         label_scores: List[float],
                         label_gold: int = None,
                         additional_metadata: Dict[str, Any] = None) -> Instance:

        fields: Dict[str, Field] = {}
        utterance_tokens = self._tokenizer.tokenize(utterance)
        prev_turns_text = ""

        if len(additional_metadata['prev_turns']) > 6:
            selected_turns = additional_metadata['prev_turns'][-6:]
        else:
            selected_turns = additional_metadata['prev_turns']

        for turn in selected_turns:
            prev_turns_text += turn + ' '

        prev_turns_text = prev_turns_text.strip()

        prev_turns_tokens = self._tokenizer.tokenize(prev_turns_text)

        source_tokens = prev_turns_tokens + utterance_tokens
        prev_turns_mask = [1]*len(prev_turns_tokens) + [0]*len(utterance_tokens)
        utt_mask = [0]*len(prev_turns_tokens) + [1]*len(utterance_tokens)

        fields["prev_turns"] = TextField(prev_turns_tokens, self._token_indexers)
        fields["curr_utt"] = TextField(utterance_tokens, self._token_indexers)

        fields['combined_source'] = TextField(source_tokens, self._token_indexers)

        fields["prev_turns_mask"] = ArrayField(np.array(prev_turns_mask))
        fields["utt_mask"] = ArrayField(np.array(utt_mask))

        fields['label_scores'] = ArrayField(np.asarray(label_scores))

        if label_gold is not None:
            fields['label_gold'] = LabelField(label_gold, skip_indexing=True)

        metadata = additional_metadata or {}
        metadata.update({'prev_turns_tokens': prev_turns_tokens,
                         'utterance_tokens': utterance_tokens,
                         'source_tokens': source_tokens})
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)
