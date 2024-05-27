import json
import re
from entity_extraction import get_gollie_entities
from summarizer import summarize_text
import nltk
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
import numpy as np

import matplotlib.pyplot as plt
import torch
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


'''This script is used to compute all the summaries with and without GoLLIE entities and compare the ROUGE and BERT scores, between the obtained results adn the 
ground truth summaries. The script also computes the average length of the summaries and saves the results in a text file. The script also saves the ROUGE scores'''

nltk.download('stopwords')
nltk.download('punkt')

CUDA_VISIBLE_DEVICES = 4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#data = json.load(open('uab_summary_2024_all_clean.json'))
gt_data = json.load(open('/hhome/nlp2_g09/Project/uab_summary_2024_with_gt.json'))
print(gt_data[0]['Summary'])
use_gollie = True
schematic = False
# Function to clean text

def filter_stopwords(text):
    stop_words = set(stopwords.words('spanish'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)


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

rouge_gollie_dict = {'rouge1':[], 'rouge2':[], 'rougeL':[]}
rouge_raw_dict = {'rouge1':[], 'rouge2':[], 'rougeL':[]}
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
        
        # Call the Groq summarization with the entities
        try:
            summary_gollie = summarize_text(text, gollie_entities, schematic=schematic)
        except:
            summary_gollie = summarize_text(text[:120], gollie_entities, schematic=schematic)
    
        
        # Call the Groq summarization without the entities
        summary_raw = summarize_text(text, schematic=schematic)
        
        
        # Initialize the ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        # Calculate ROUGE scores
        no_stopwords_gollie = filter_stopwords(summary_gollie)
        no_stopwords_raw = filter_stopwords(summary_raw)
        scores_gollie = scorer.score(no_stopwords_gollie,gt_summary)
        scores_raw = scorer.score(no_stopwords_raw,gt_summary)
        
        print("ROUGE scores gollie:")
        for score_type, score_values in scores_gollie.items():
            print(f"{score_type}: {score_values}")
            rouge_gollie_dict[score_type].append(score_values)
        
            
            
        print("ROUGE scores raw:")
        for score_type, score_values in scores_raw.items():
            print(f"{score_type}: {score_values}")
            rouge_raw_dict[score_type].append(score_values)
        
        
        len_gollie.append(len(summary_gollie))
        len_raw.append(len(summary_raw))
            
        # Open a file in write mode
        with open("results_no_stopwords.txt", "a") as file:
        
            # Your print statements
            # Write a string to the file
            new_text = f"\nInput Text:{text}\n"
            file.write(new_text)
            # Write another line
            new_text = f"\nGT Summary: {gt_summary}"
            file.write(new_text)
            new_text = f"\nGOLLIE Summary: {summary_gollie}"
            file.write(new_text)
            file.write("\nROUGE scores gollie:")
            for score_type, score_values in scores_gollie.items():
                new_text = f"\n{score_type}: {score_values}"
                file.write(new_text)
                
            new_text = f"\nRAW Summary: {summary_raw}"    
            file.write(new_text)
            file.write("\nROUGE scores raw:")
            for score_type, score_values in scores_raw.items():
                new_text = f"\n{score_type}: {score_values}"
                file.write(new_text)
            
            



# Get the average scores for each metric
rouge1_score_gollie = np.mean([samp for samp in rouge_gollie_dict['rouge1'] if samp != 0])
rouge2_score_gollie= np.mean([samp for samp in rouge_gollie_dict['rouge2'] if samp != 0])
rougeL_score_gollie = np.mean([samp for samp in rouge_gollie_dict['rougeL'] if samp != 0])

rouge1_score_raw = np.mean([samp for samp in rouge_raw_dict['rouge1'] if samp != 0])
rouge2_score_raw= np.mean([samp for samp in rouge_raw_dict['rouge2'] if samp != 0])
rougeL_score_raw = np.mean([samp for samp in rouge_raw_dict['rougeL'] if samp != 0])

print("Average ROUGE scores gollie: \n","ROUGE1: ", rouge1_score_gollie, "ROUGE2: ",rouge2_score_gollie,"ROUGE L: ", rougeL_score_gollie)
print("Average length of summaries gollie: ", np.mean(len_gollie))


print("\nAverage ROUGE scores raw: \n","ROUGE1: ", rouge1_score_raw , "ROUGE2: ",rouge2_score_raw ,"ROUGE L: ",rougeL_score_raw )
print("Average length of summaries raw: ", np.mean(len_raw))

# Open a file in write mode
with open("results_no_stopwords.txt", "a") as file:
    new_text = f"\nAverage ROUGE scores gollie:\n ROUGE1:{rouge1_score_gollie} \nROUGE2: {rouge2_score_gollie}\nROUGE L: {rougeL_score_gollie}"
    file.write(new_text)
    new_text = f"\nAverage ROUGE scores raw:\n ROUGE1:{rouge1_score_raw} \nROUGE2: {rouge2_score_raw}\nROUGE L: {rougeL_score_raw}"
    file.write(new_text)
    
    new_text = f"\nAverage length of summaries gollie: {np.mean(len_gollie)}"
    file.write(new_text)
    new_text = f"\nAverage length of summaries raw: {np.mean(len_raw)}"
    file.write(new_text)
    
X = ['rouge1', 'rouge2', 'rougeL'] 
gollie_res = [rouge1_score_gollie,rouge2_score_gollie,rougeL_score_gollie] 
raw_res = [rouge1_score_raw,rouge2_score_raw,rougeL_score_raw] 
  
X_axis = np.arange(len(X)) 
  
plt.bar(X_axis - 0.2, gollie_res, 0.4, label = 'Gollie') 
plt.bar(X_axis + 0.2, raw_res, 0.4, label = 'Raw') 
  
plt.xticks(X_axis, X) 
plt.xlabel("Groups") 
plt.ylabel("Score") 
plt.title("Rouge Scores") 
plt.legend() 
plt.savefig("results_no_stopwords.png")
plt.show() 


