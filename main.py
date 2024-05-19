import PyPDF2
import spacy
import json

from loguru import logger
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer

SMALL_NLP_NM = 'en_core_web_sm'
MEDIUM_NLP_NM = 'en_core_web_md'
LARGE_NLP_NM = 'en_core_web_lg'

# Function to extract text from PDF
## Deprecated in PyPDF2 3.0.0: PyPDF2.PdfFileReader, etc.
# def extract_text_from_pdf(pdf_path):
#     pdf_file_obj = open(pdf_path, 'rb')
#     pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
#     text = ''
#     for page_num in range(pdf_reader.numPages):
#         page_obj = pdf_reader.getPage(page_num)
#         text += page_obj.extractText()
#     pdf_file_obj.close()
#     return text

##
##
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file_obj:
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

##
##
def load_paragraphs(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

##
##
def save_to_json(results, output_path='output.json'):
    with open(output_path, 'w') as f:
        json.dump(results, f)

##
##
def chunk_similarity(paragraphs, pdf_text, nlp):
    results = []
    for paragraph in paragraphs:
        doc1 = nlp(paragraph['text'])
        paragraph_sentences = list(doc1.sents)
        num_sentences_in_paragraph = len(paragraph_sentences)
        pdf_sentences = list(nlp(pdf_text).sents)
        max_similarity = 0
        matched_text = ''
        for i in range(len(pdf_sentences) - num_sentences_in_paragraph + 1):
            chunk = pdf_sentences[i:i+num_sentences_in_paragraph]
            chunk_text = ' '.join([sent.text for sent in chunk])
            doc2 = nlp(chunk_text)
            similarity = doc1.similarity(doc2)
            if similarity > max_similarity:
                max_similarity = similarity
                matched_text = chunk_text
        if max_similarity > 0.65:
            results.append({
                'name': paragraph['name'],
                'text': paragraph['text'],
                'matched': matched_text,
                'similarity': max_similarity
            })
        else:
            results.append({
                'name': paragraph['name'],
                'text': paragraph['text'],
                'matched': '',
                'similarity': 0.0
            })

##
##
def simple_similarity(paragraphs, pdf_text, nlp):
    results = []
    for paragraph in paragraphs:
        doc1 = nlp(paragraph['text'])
        max_similarity = 0
        matched_text = ''
        for sent in nlp(pdf_text).sents:
            doc2 = nlp(sent.text)
            similarity = doc1.similarity(doc2)
            if similarity > max_similarity:
                max_similarity = similarity
                matched_text = sent.text
        if max_similarity > 0.65:
            results.append({
                'name': paragraph['name'],
                'text': paragraph['text'],
                'matched': matched_text,
                'similarity': max_similarity
            })
        else:
            results.append({
                'name': paragraph['name'],
                'text': paragraph['text'],
                'matched': '',
                'similarity': 0.0
            })
    return results


##
##
### ----------------- MAIN ----------------- ###
##
##
if __name__ == '__main__':
    logger.debug('------ Start! ------')
    model = MEDIUM_NLP_NM
    logger.debug(f'Loading {model} NLP model...')
    nlp = spacy.load(model)
    logger.debug('Loading PDF text')
    pdf_text = extract_text_from_pdf('data/long_text.pdf')
    logger.debug('Loading paragraphs...')
    paragraphs = load_paragraphs('config/paragraphs.json')
    logger.debug('Calculating SIMPLE similarity...')
    results = simple_similarity(paragraphs, pdf_text, nlp)
    logger.debug('\nResults: -----------------')
    logger.debug(results)
    logger.debug('\nCalculating CHUNK similarity...')
    results = chunk_similarity(paragraphs, pdf_text, nlp)
    logger.debug('\nResults: -----------------')
    logger.debug(results)
    logger.debug('-------- End! --------')

####  ----------------------  ####