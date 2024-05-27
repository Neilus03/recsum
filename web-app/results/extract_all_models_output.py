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



CUDA_VISIBLE_DEVICES = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#data = json.load(open('uab_summary_2024_all_clean.json'))
gt_data = json.load(open('/hhome/nlp2_g09/Project/uab_summary_2024_with_gt.json'))
print(gt_data[0]['Summary'])
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



def compute_rouge_and_bert(gt_data):
    output_file = {"gollie_summaries": {"Llama3-70b-8192":[], "Llama3-8b-8192":[], "Mixtral-8x7b-32768":[]}, "raw_summaries":{'Gemma-7b-It':[], "Llama3-70b-8192":[], "Llama3-8b-8192":[], "Mixtral-8x7b-32768":[]} }
    len_gollie = []
    len_raw = []


    for gt in gt_data:
        gt_summary = gt['Summary']
        gt_text = gt['Text']
        
        text = clean_text(gt_text)
        
        if use_gollie:
            # Divide the text into patches
            text_patches = divide_text(text)
            
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
                f.write(text + '\n\n')
                f.write(gollie_entities)
            for model in output_file['gollie_summaries'].keys():
                # Call the Groq summarization with the entities
                try:
                    summary_gollie = summarize_text(text, model, gollie_entities, schematic=schematic)
                    output_file['gollie_summaries'][model].append(summary_gollie)
                except:
                    summary_gollie = summarize_text(text[:600],model, gollie_entities, schematic=schematic)
                    output_file['gollie_summaries'][model].append(summary_gollie)
            
                # Call the Groq summarization without the entities
            
                summary_raw = summarize_text(text,model, schematic=schematic)
                output_file['raw_summaries'][model].append(summary_raw)
            
    # Specify the file path where you want to save the JSON data
    file_path = 'models_output.json'

    # Open the file in write mode and use json.dump to write the dictionary to the file
    with open(file_path, 'w') as json_file:
        json.dump(output_file, json_file, indent=4, ensure_ascii=False)        
  



