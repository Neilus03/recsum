from compute_density import compute_density_main
from compute_keyword_accuracy import keyword_accuracy_main
from extract_all_models_output import compute_rouge_and_bert
import json
import numpy as np
import torch 


CUDA_VISIBLE_DEVICES = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#data = json.load(open('uab_summary_2024_all_clean.json'))
gt_data = json.load(open('/hhome/nlp2_g09/recsum/web-app/main/results/inicial_keyword.json'))
gt_data_texts = gt_data['textos']
use_gollie = True
schematic = False


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
    


CUDA_VISIBLE_DEVICES = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#data = json.load(open('uab_summary_2024_all_clean.json'))
gt_data = json.load(open('/hhome/nlp2_g09/Project/uab_summary_2024_with_gt.json'))
print(gt_data[0]['Summary'])
use_gollie = True
schematic = False

    
compute_rouge_and_bert(gt_data)
    
    
