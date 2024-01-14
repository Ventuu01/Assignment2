import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import pdfplumber
import requests
import io

# Imposta la massima lunghezza del contesto e la dimensione dell'overlap per l'elaborazione del testo
max_context_window = 2048  # Ridotto per dimostrazione
overlap_size = 60  # Sovrapposizione tra segmenti
cosine_threshold = 0.2 

# Funzione per il pre-processing del testo
def preprocess_text_basic(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # conversione testo in minuscolo
    text = ''.join([char for char in text if char not in string.punctuation])  # rimozione punteggiatura
    words = nltk.word_tokenize(text)  # Tokenizzazione testo
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatizzazione e rimozione stopwords
    preprocessed_text = ' '.join(lemmatized_words)  # ricostruzione testo
    token_count = len(lemmatized_words)  # Conteggio token
    return preprocessed_text, token_count

# Funzione per dividere il testo in segmenti
def split_text(text, slice_size, overlap):
    slices = []
    start = 0
    while start < len(text):
        end = start + slice_size
        slice_text = text[start:end]
        slices.append(slice_text)
        start = end - overlap
    return slices

# Funzione per calcolare la distanza del coseno (similarità) tra due testi
def cosine_distance(text1, text2):
    vectorizer = CountVectorizer().fit([text1, text2])  # Creazione vettore di parole
    vectors = vectorizer.transform([text1, text2])
    return cosine_similarity(vectors)[0][1]

# Funzione per elaborare il testo di input
def process_text_basic(input_text):
    preprocessed_text, num_tokens = preprocess_text_basic(input_text)
    print(f"N: di token: {num_tokens}")
    if num_tokens <= max_context_window:
        return [preprocessed_text], []  # Nessuna divisione se il testo è breve
    else:
        # Divide il testo in segmenti
        slices = split_text(preprocessed_text, max_context_window, overlap_size)
        
        # Filtra i segmenti in base alla similarità del coseno
        filtered_slices = []
        similarities = []
        for i in range(len(slices) - 1):
            similarity = cosine_distance(slices[i], slices[i + 1])
            similarities.append(similarity)
            if similarity < cosine_threshold:
                filtered_slices.append(slices[i])
        filtered_slices.append(slices[-1]) 

        return filtered_slices, similarities


#Funzione per la lettura del file pdf
def read_text_from_pdf_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        pdf_file = io.BytesIO(response.content)
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + " "
        return text
    else:
        raise Exception(f"Failed to download file from {url}")

# Percorso del file PDF
pdf_url = 'https://muse.jhu.edu/pub/4/oa_monograph/chapter/2566849'
input_text_from_pdf = read_text_from_pdf_url(pdf_url)


# Elaborazione testo
processed_slices_basic, similarities = process_text_basic(input_text_from_pdf)

# Stampa segmenti e similarità con il segmento successivo
for i, (slice, similarity) in enumerate(zip(processed_slices_basic, similarities + [None])):
    print(f"Segmento {i+1}:")
    print(slice)
    print(f"Similarità con il segmento successivo: {similarity}")
    print("-----------------------------------------------------")