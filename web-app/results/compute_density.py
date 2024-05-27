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
import re


CUDA_VISIBLE_DEVICES = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#data = json.load(open('uab_summary_2024_all_clean.json'))
gt_data = json.load(open('/hhome/nlp2_g09/recsum/web-app/main/results/inicial_keyword.json'))
gt_data_texts = gt_data['textos']
use_gollie = True
schematic = False


# Function to clean text
def clean_text(text):
    # Remove unwanted characters and symbols
    text = re.sub(r'[\u00a0\u00ad\u200b\u200c\u200d\u202a-\u202e]', '', text)  # Removing non-breaking spaces and soft hyphens
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(r'[*]', '', text)  #remove * characters
    return text.strip()

def divide_text(text, n=120):
    # Regular expression to split the text into sentences
    sentence_endings = re.compile(r'(?<=[.!?]) +') # Split by '.', '!' or '?'
    sentences = sentence_endings.split(text)
    
    # Join sentences until the number of words exceeds 'n'
    result = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words = sentence.split()
        current_length += len(words)
        current_chunk.append(sentence)
        
        if current_length > n:
            result.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
    
    # Add the last chunk if it exists
    if current_chunk:
        result.append(' '.join(current_chunk))
    
    return result


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
                summary_gollie = summarize_text(gt_text[:600],model, gollie_entities, schematic=schematic)
                output_summaries['gollie_summaries'][model].append(summary_gollie)

            # Call the Groq summarization without the entities
            try:
                summary_raw = summarize_text(gt_text,model, schematic=schematic)
                output_summaries['raw_summaries'][model].append(summary_raw)
            except:
                summary_raw = summarize_text(gt_text[:600],model, schematic=schematic)
                output_summaries['raw_summaries'][model].append(summary_raw)
            
            
            
            
    return output_summaries

def compute_density_summary(pred_summary, keywords):
    
    keywords_len = [len(keyword.split()) for keyword in keywords]
    mean_keyword_len = np.mean(keywords_len)
    '''for i in range(max_keyword_len, 0, -1):
        count_min = 0 
        count_max = 0 
        for _ in range(int(len(pred_summary.split())/i)):
            count_max += i 
            if keywords in pred_summary.split()[count_min:count_max]:
                keywords_count += 1
            count_max'''
    pattern = '|'.join(re.escape(keyword) for keyword in keywords)
    # Find all matches in the text
    matches = re.findall(pattern, pred_summary.lower())

    # Count the number of matches
    keyword_count = len(matches)
    return keyword_count /len(pred_summary.split())
        
            
def compute_density_main(texts):
    density = {'gollie_densities':{'Gemma-7b-It':[],"Llama3-70b-8192":[], "Llama3-8b-8192":[], "Mixtral-8x7b-32768":[]}, 'raw_densities':{'Gemma-7b-It':[],"Llama3-70b-8192":[], "Llama3-8b-8192":[], "Mixtral-8x7b-32768":[]}}
    for text in texts:
        gt_text = text['texto']
        keywords = text['keywords']
        gt_text = clean_text(gt_text)
        output_summaries = compute_summaries(gt_text)
        for model in output_summaries['gollie_summaries'].keys():
            for summary in output_summaries['gollie_summaries'][model]:
                density['gollie_densities'][model].append(compute_density_summary(summary, keywords))
        for model in output_summaries['raw_summaries'].keys():
            for summary in output_summaries['raw_summaries'][model]:
                density['raw_densities'][model].append(compute_density_summary(summary, keywords))
    return density


    

        
        
density_dict = compute_density_main(gt_data_texts)
avg_density_dict = {"average_gollie_density":{"Gemma-7b-It":np.mean(density_dict['gollie_densities']["Gemma-7b-It"]),
                                              "Llama3-70b-8192":np.mean(density_dict['gollie_densities']["Llama3-70b-8192"]), 
                                               "Llama3-8b-8192":np.mean(density_dict['gollie_densities']["Llama3-8b-8192"]), 
                                               "Mixtral-8x7b-32768":np.mean(density_dict['gollie_densities']["Mixtral-8x7b-32768"])},
                    "average_raw_density":{"Gemma-7b-It":np.mean(density_dict['raw_densities']["Gemma-7b-It"]),
                                           "Llama3-70b-8192":np.mean(density_dict['raw_densities']["Llama3-70b-8192"]), 
                                            "Llama3-8b-8192":np.mean(density_dict['raw_densities']["Llama3-8b-8192"]), 
                                            "Mixtral-8x7b-32768":np.mean(density_dict['raw_densities']["Mixtral-8x7b-32768"])}}


for model in density_dict['gollie_densities'].keys():
    print("Average GoLLIE Density for ", model, ": ", np.mean(density_dict['gollie_densities'][model]))
    print("Average Raw Density for ", model, ": ", np.mean(density_dict['raw_densities'][model]))


# Specify the file path where you want to save the JSON data
file_path = '/hhome/nlp2_g09/recsum/web-app/main/results/densities_gemma.json'

# Open the file in write mode and use json.dump to write the dictionary to the file
with open(file_path, 'w') as json_file:
    json.dump(density_dict, json_file, indent=4, ensure_ascii=False)   
    
# Specify the file path where you want to save the JSON data
file_path1 = '/hhome/nlp2_g09/recsum/web-app/main/results/avg_densities_gemma.json'

# Open the file in write mode and use json.dump to write the dictionary to the file
with open(file_path1, 'w') as json_file:
    json.dump(avg_density_dict, json_file, indent=4, ensure_ascii=False)      