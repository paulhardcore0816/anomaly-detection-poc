import os
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def normalize_json(json_data):
    """Normalize JSON for comparison by sorting keys."""
    return json.dumps(json_data, sort_keys=True)

def compute_similarity(json_files):
    """Compute similarity scores between JSON files."""
    normalized_files = []
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        normalized_files.append(normalize_json(data))
    
    # Vectorize the normalized JSON files
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(normalized_files)
    
    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def detect_anomaly(json_files):
    """Detect the anomalous file based on similarity scores."""
    similarity_matrix = compute_similarity(json_files)
    avg_similarity = similarity_matrix.mean(axis=1)
    
    # Find the file with the lowest average similarity
    anomaly_index = np.argmin(avg_similarity)
    anomaly_file = json_files[anomaly_index]
    return anomaly_file, avg_similarity



# Load all JSON files from the "data" folder
data_folder = "data"  # Replace with your folder name
json_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.json')]

print(json_files)

# Detect the anomaly
if json_files:
    anomaly_file, avg_similarity = detect_anomaly(json_files)
    print(f"Anomalous file detected: {anomaly_file}")
    print("Average Similarity Scores:", avg_similarity)
else:
    print("No JSON files found in the specified folder.")

