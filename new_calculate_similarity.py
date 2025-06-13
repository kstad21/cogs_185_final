import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REFERENCE_MAP = {
    "comedy": "divine_comedy.txt",
    "shakespeare": "shakespeare.txt",
    "paradise": "paradise_lost.txt",
    "iliad": "iliad.txt",
}

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

def main():
    results_dir = "./results"
    data_dir = "./old_data"
    output_csv = "generic_similarity_results.csv"

    # Preload all reference texts to avoid reloading them for each generated file
    reference_texts = {}
    for key, filename in REFERENCE_MAP.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            reference_texts[key] = read_file(path)
        else:
            print(f"[!] Missing reference file: {path}")

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['generated_file'] + [f"{key}_similarity" for key in REFERENCE_MAP.keys()]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result_file in os.listdir(results_dir):
            if not result_file.endswith(".txt"):
                continue

            result_path = os.path.join(results_dir, result_file)
            result_text = read_file(result_path)

            row = {'generated_file': result_file}

            for key, ref_text in reference_texts.items():
                similarity = calculate_cosine_similarity(result_text, ref_text)
                row[f"{key}_similarity"] = similarity

            writer.writerow(row)
            print(f"[✓] Processed {result_file}")

if __name__ == "__main__":
    main()
