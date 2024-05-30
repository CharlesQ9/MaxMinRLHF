import os
os.environ['OPENAI_API_KEY'] = "" #TODO: openai key
from alpaca_farm.utils import jload
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
from alpaca_eval.metrics import pairwise_to_winrate
import json

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)

output_path1="MaxMinRLHF/llama2_output/llama2_P2B_P2_10_1.json"#TODO:change path
output_path2="MaxMinRLHF/llama2_output/llama2_P2B_baseline.json"#TODO:change path

outputs1 = jload(output_path1)
outputs2 = jload(output_path2)

outputs1_new=[]
outputs2_new=[]
for i in range(len(outputs1)):
    if outputs1[i]["output"]=="":
        continue
    if outputs2[i]["output"]=="":
        continue
    outputs1_new.append(outputs1[i])
    outputs2_new.append(outputs2[i])

annotator_pool = PairwiseAutoAnnotator()
annotated_sft = annotator_pool.annotate_head2head(outputs_1=outputs1_new, outputs_2=outputs2_new)
print(annotated_sft)
print(pairwise_to_winrate(preferences=[a["preference"] for a in annotated_sft]))
