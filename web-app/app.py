from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import sys
import subprocess
import fitz  # PyMuPDF
import re
from main.entity_extraction_2 import get_gollie_entities
from whisper_web import whisper_transcribe

# Add the GoLLIE_MED/src directory to the sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from main.summarizer import summarize_text

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables
gollie_entities = []

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Ensure the main directory exists
if not os.path.exists('text'):
    os.makedirs('text')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = ""
    use_gollie = request.form.get('use_gollie') == 'true'
    schematic = request.form.get('schematic') == 'true'
    
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.rsplit('.', 1)[1].lower() == 'pdf':
                text = extract_text_from_pdf(filepath)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
        else:
            return jsonify({'error': 'File type not allowed'})
    elif 'text' in request.form:
        text = request.form['text']
    else:
        return jsonify({'error': 'No valid input provided'})

    # Write the text to a temporary file
    with open('text/input_text.txt', 'w') as f:
        f.write(text)

    text = clean_text(text)
    
    if use_gollie:
        # Divide the text into patches
        text_patches = divide_text(text)
        
        # Write the patches to a temporary file
        with open('text/input_text_patches.txt', 'w') as f:
            f.write('\n\n\n'.join(text_patches))
        
        gollie_entities = []
        
        for patch in text_patches:
            entities = get_gollie_entities(patch)
            if len(entities) > 0:
                gollie_entities.append(entities)
        
        gollie_entities = "\n".join(gollie_entities)

        # Write the entity extraction output to a temporary file
        with open('text/text_and_entities.txt', 'w') as f:
            f.write(text + '\n\n')
            f.write(gollie_entities)
        
        # Call the Groq summarization with the entities
        summary = summarize_text(text, gollie_entities, schematic=schematic)
    
        '''# Read the entity extraction output
        with open('main/gollie_output.txt', 'r') as f:
            gollie_output = f.read()'''

    else:
        # Call the Groq summarization without the entities
        summary = summarize_text(text, schematic=schematic)

    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True, port=5080)
