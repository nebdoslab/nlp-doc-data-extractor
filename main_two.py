import PyPDF2
import spacy
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_file_obj = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
    text = ''
    for page_num in range(pdf_reader.numPages):
        page_obj = pdf_reader.getPage(page_num)
        text += page_obj.extractText()
    pdf_file_obj.close()
    return text

# Function to load paragraphs from JSON
def load_paragraphs(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Load NLP model
nlp = spacy.load('en_core_web_sm')

# Load PDF text and paragraphs
pdf_text = extract_text_from_pdf('data/long_text.pdf')
paragraphs = load_paragraphs('config/paragraphs.json')

# Vectorize the text
vectorizer = TfidfVectorizer().fit_transform([pdf_text] + [p['text'] for p in paragraphs])

# Find similar sections and calculate similarity scores
results = []
for i, paragraph in enumerate(paragraphs, start=1):
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[i:i+1])
    results.append({
        'name': paragraph['name'],
        'similar_sections': [{'text': chunk.text, 'similarity_score': similarities[0][0]} for chunk in nlp(pdf_text).noun_chunks]
    })

# Save output to JSON
with open('output.json', 'w') as f:
    json.dump(results, f)