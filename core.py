import os
import re
import spacy
import csv
import uuid
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from urllib.parse import urljoin
import face_recognition
from deepface import DeepFace

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Folder paths
dataset_folder = "input_images"
temp_folder = "temp_images"
os.makedirs(temp_folder, exist_ok=True)

# Load crime-related keywords
def load_crime_keywords(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        return [row[0].strip() for row in reader]

crime_keywords = load_crime_keywords('crime_keywords.csv')

# Classify article text as crime or not
def classify_text(text):
    pattern = re.compile(r'\b(?:' + '|'.join(crime_keywords) + r')\b', re.IGNORECASE)
    if pattern.search(text):
        return "criminal"
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE"] and any(keyword in ent.text.lower() for keyword in crime_keywords):
            return "criminal"
    return "not criminal"

# Extract text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([para.get_text() for para in paragraphs])
    except:
        return None

# Extract all image URLs from article
def extract_images_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return [urljoin(url, img["src"]) for img in soup.find_all("img") if "src" in img.attrs]
    except:
        return []

# Download and save images locally
def download_images(image_urls, temp_folder):
    image_paths = []
    for url in image_urls:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            path = os.path.join(temp_folder, f"{uuid.uuid4().hex}.jpg")
            img.save(path)
            image_paths.append(path)
        except:
            continue
    return image_paths

# Encode known face images from the dataset
def encode_dataset(folder):
    encodings, names = [], []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            img = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(img)
            if enc:
                encodings.append(enc[0])
                names.append(file)
        except:
            continue
    return encodings, names

# Match image faces against known faces
def find_culprit(encodings, names, images):
    for path in images:
        try:
            img = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(img)
            if not enc:
                continue
            for face in enc:
                matches = face_recognition.compare_faces(encodings, face)
                if any(matches):
                    return f"Matched with known criminal: {names[matches.index(True)]}"
        except:
            continue
    return "Culprit not found."

# Analyze emotions using DeepFace
def analyze_emotions(images):
    suspects = []
    for img in images:
        try:
            res = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
            if isinstance(res, list):
                res = res[0]
            emotion = res.get("dominant_emotion", "neutral")
            if emotion in ["angry", "fear", "disgust"]:
                suspects.append((img, emotion))
        except:
            continue
    return suspects

# Delete temp images
def cleanup(images):
    for path in images:
        try:
            os.remove(path)
        except:
            pass

# Main function called by app.py
def process_url(url):
    text = extract_text_from_url(url)
    if not text:
        return {"error": "Could not extract article text."}

    classification = classify_text(text)
    if classification != "criminal":
        return {"result": "This article is not crime-related."}

    images = extract_images_from_url(url)
    if not images:
        return {"result": "No images found in article."}

    paths = download_images(images, temp_folder)
    if not paths:
        return {"result": "Could not download any valid images."}

    enc, names = encode_dataset(dataset_folder)
    culprit = find_culprit(enc, names, paths)
    if "Matched" in culprit:
        cleanup(paths)
        return {"result": culprit}

    suspects = analyze_emotions(paths)
    cleanup(paths)
    if suspects:
        return {"result": f"Suspicious behavior detected (Emotion: {suspects[0][1]})"}

    return {"result": "No criminal or suspicious behavior detected."}
