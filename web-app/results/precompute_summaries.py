import json
import re
from entity_extraction import get_gollie_entities
from summarizerAllModels import summarize_text
import nltk
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
import numpy as np

import matplotlib.pyplot as plt
import torch
import bert_score
from compute_density import clean_text, divide_text, compute_summaries



CUDA_VISIBLE_DEVICES = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#data = json.load(open('uab_summary_2024_all_clean.json'))
gt_data = json.load(open('/hhome/nlp2_g09/recsum/web-app/main/results/inicial_keyword.json'))
gt_data_texts = gt_data['textos']
use_gollie = True
schematic = False


output_dict = {'textos':[]}

for text in gt_data_texts:
    gt_text = text['texto']
    keywords = text['keywords']
    gt_text = clean_text(gt_text)
    output_summaries = compute_summaries(gt_text)
    new_dict = dict()
    new_dict['texto'] = gt_text
    new_dict['keywords'] = keywords
    new_dict['gollie_summaries'] = output_summaries['gollie_summaries']
    new_dict['raw_summaries'] = output_summaries['raw_summaries']
    
    output_dict['textos'].append(new_dict)
    
# Specify the file path where you want to save the JSON data
file_path = '/hhome/nlp2_g09/recsum/web-app/main/results/results_outputs/precomputed_summaries.json'

# Open the file in write mode and use json.dump to write the dictionary to the file
with open(file_path, 'w') as json_file:
    json.dump(output_dict, json_file, indent=4, ensure_ascii=False) 