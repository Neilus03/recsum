# Towards Improved Recall in Medical DocumenT Summarization: The GoLLIE Approach

![Summarization Pipeline](https://github.com/Neilus03/recsum/assets/87651732/de670f6e-720b-4d0d-821c-0531151e15cb)


This repository contains the code and resources for the Medical Document Summarizer, which uses the GoLLIE approach to improve recall in medical document summarization. The summarizer extracts key entities and details from medical texts and generates structured summaries using a few-shot learning approach on Llama3.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Repository Structure](#repository-structure)
6. [Experiments and Results](#experiments-and-results)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)

## Introduction

Maintaining the accuracy of extracted information in medical document summarization is crucial due to the potential consequences of errors. This project leverages GoLLIE, a Guideline-following Large Language Model for Information Extraction, to enhance recall by identifying key entities and essential details in medical texts. The extracted entities and details are used to generate structured summaries using a few-shot learning approach on Llama3.

For more details, refer to our [paper](https://github.com/Neilus03/recsum/blob/main/Towards_Improved_Recall_in_Medical_Document_Summarization%3AThe_GoLLIE_Approach.pdf).

## Features

- **Accurate Information Extraction**: Uses GoLLIE to extract key entities and essential details from medical texts.
- **Structured Summarization**: Generates concise and informative summaries using Llama3.
- **Few-shot Learning**: Eliminates the need for extensive retraining of the summarizing LLM.
- **Multilingual Support**: Supports extraction of information from Spanish medical reports as well.

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
