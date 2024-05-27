
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
from compute_density import clean_text, divide_text



CUDA_VISIBLE_DEVICES = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#data = json.load(open('uab_summary_2024_all_clean.json'))
gt_data = json.load(open('/hhome/nlp2_g09/recsum/web-app/main/results/inicial_keyword.json'))
gt_data_texts = gt_data['textos']
use_gollie = True
schematic = False



def compute_summaries(gt_text):
    output_summaries=  {"gollie_summaries": {'Gemma-7b-It':[],"Llama3-70b-8192":[], "Llama3-8b-8192":[], "Mixtral-8x7b-32768":[]}, "raw_summaries":{'Gemma-7b-It':[], "Llama3-70b-8192":[], "Llama3-8b-8192":[], "Mixtral-8x7b-32768":[]} }
    if use_gollie:
        # Divide the text into patches
        text_patches = divide_text(gt_text)
        
        # Write the patches to a temporary file
        with open('/hhome/nlp2_g09/recsum/web-app/main/text/text_and_entities.txt', 'a') as f:
            f.write('\n\n\n'.join(text_patches))
        
        gollie_entities = []
        
        for patch in text_patches:
            entities = get_gollie_entities(patch)
            if len(entities) > 0:
                gollie_entities.append(entities)
        
        gollie_entities = "\n".join(gollie_entities)

        # Write the entity extraction output to a temporary file
        with open('/hhome/nlp2_g09/recsum/web-app/main/text/text_and_entities.txt', 'a') as f:
            f.write(gt_text + '\n\n')
            f.write(gollie_entities)
        for model in output_summaries['gollie_summaries'].keys():
            # Call the Groq summarization with the entities
            try:
                summary_gollie = summarize_text(gt_text, model, gollie_entities, schematic=schematic)
                output_summaries['gollie_summaries'][model].append(summary_gollie)
            except:
                summary_gollie = summarize_text(gt_text[:1000],model, gollie_entities, schematic=schematic)
                output_summaries['gollie_summaries'][model].append(summary_gollie)

            # Call the Groq summarization without the entities
            try:
                summary_raw = summarize_text(gt_text,model, schematic=schematic)
                output_summaries['raw_summaries'][model].append(summary_raw)
            except:
                summary_raw = summarize_text(gt_text[:1000],model, schematic=schematic)
                output_summaries['raw_summaries'][model].append(summary_raw)
              
    return output_summaries

def compute_keyword_accuracy(summary,keywords):
    pattern = '|'.join(re.escape(keyword) for keyword in keywords)
    # Find all matches in the text
    matches = re.findall(pattern, summary.lower())

    # Count the number of matches
    keyword_count = len(matches)
    return keyword_count/len(keywords)

def keyword_accuracy_main(texts):
    keyword_accuracy = {'gollie_acc':{'Gemma-7b-It':[],"Llama3-70b-8192":[], "Llama3-8b-8192":[], "Mixtral-8x7b-32768":[]}, 'raw_acc':{'Gemma-7b-It':[],"Llama3-70b-8192":[], "Llama3-8b-8192":[], "Mixtral-8x7b-32768":[]}}
    for text in texts:
        gt_text = text['texto']
        keywords = text['keywords']
        gt_text = clean_text(gt_text)
        output_summaries = compute_summaries(gt_text)
        for model in output_summaries['gollie_summaries'].keys():
            for summary in output_summaries['gollie_summaries'][model]:
                acc_gol = compute_keyword_accuracy(summary, keywords)
                if acc_gol != 0:
                    keyword_accuracy['gollie_acc'][model].append(acc_gol)
        for model in output_summaries['raw_summaries'].keys():
            for summary in output_summaries['raw_summaries'][model]:
                acc_raw = compute_keyword_accuracy(summary, keywords)
                if acc_raw != 0:
                    keyword_accuracy['raw_acc'][model].append(acc_raw)
                
    return keyword_accuracy


density_dict = keyword_accuracy_main(gt_data_texts)

avg_density_dict = {"average_gollie_accuracy":{"Gemma-7b-It":np.mean(density_dict['gollie_acc']["Gemma-7b-It"]),
                                               "Llama3-70b-8192":np.mean(density_dict['gollie_acc']["Llama3-70b-8192"]), 
                                               "Llama3-8b-8192":np.mean(density_dict['gollie_acc']["Llama3-8b-8192"]), 
                                               "Mixtral-8x7b-32768":np.mean(density_dict['gollie_acc']["Mixtral-8x7b-32768"])}, 
                    "average_raw_accuracy":{"Gemma-7b-It":np.mean(density_dict['raw_acc']["Gemma-7b-It"]),
                                            "Llama3-70b-8192":np.mean(density_dict['raw_acc']["Llama3-70b-8192"]), 
                                            "Llama3-8b-8192":np.mean(density_dict['raw_acc']["Llama3-8b-8192"]), 
                                            "Mixtral-8x7b-32768":np.mean(density_dict['raw_acc']["Mixtral-8x7b-32768"])}}

for model in avg_density_dict['average_gollie_accuracy'].keys():
    print("Average GoLLIE Accuracy for ", model, ": ", avg_density_dict['average_gollie_accuracy'][model])
    print("Average Raw Accuracy for ", model, ": ", avg_density_dict['average_raw_accuracy'][model])


# Specify the file path where you want to save the JSON data
file_path = '/hhome/nlp2_g09/recsum/web-app/main/results/results_outputs/keyword_acc_gemma.json'

# Open the file in write mode and use json.dump to write the dictionary to the file
with open(file_path, 'w') as json_file:
    json.dump(density_dict, json_file, indent=4, ensure_ascii=False)   
    
# Specify the file path where you want to save the JSON data
file_path1 = '/hhome/nlp2_g09/recsum/web-app/main/results/results_outputs/avg_acc_gemma.json'

# Open the file in write mode and use json.dump to write the dictionary to the file
with open(file_path1, 'w') as json_file:
    json.dump(avg_density_dict, json_file, indent=4, ensure_ascii=False) 