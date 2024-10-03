from flask import Flask, render_template, request
import numpy as np
import math
import os

static_folder_path = os.path.abspath('./static')
app = Flask(__name__, static_folder=static_folder_path)

# Define the path to your documents
document_files_path = r'static\final_project_dataset'
document_image_folder = 'images'  # Assuming your images are stored here

original_webpage_links =[
    "https://www.bbc.com/news/articles/c51ylvl8rrlo",
    "https://thehimalayantimes.com/science-and-tech/astronomers-find-what-may-be-the-universes-brightest-object-with-a-black-hole-devouring-a-sun-a-day",
    "https://www.bbc.com/news/articles/cy76gx0rgzxo",
    "https://thehimalayantimes.com/kathmandu/heavy-rainfall-expected-in-four-provinces-today",
    "https://www.bbc.com/news/articles/cvg3r35j04po",
    "https://indianexpress.com/article/technology/tech-news-technology/meta-regulatory-heat-australia-user-data-ai-models-9563809/",
    "https://www.bbc.com/news/articles/cz5reve8476o",
    "https://www.bbc.com/news/articles/c93pr5ewr22o",
    "https://thehimalayantimes.com/science-and-tech/openai-reveals-sora-a-tool-to-make-instant-videos-from-written-prompts",
    "https://thehimalayantimes.com/science-and-tech/trio-wins-nobel-prize-in-chemistry-for-work-on-quantum-dots-used-in-electronics-and-medical-imagingv",
]

# Reading document files
document_files = os.listdir(document_files_path)
docs = []
document_metadata = []  # List to store metadata for each document

for i, filename in enumerate(document_files):
    file_path = os.path.join(document_files_path, filename)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        docs.append(content)
        
        # Extract the first line as the title
        lines = content.splitlines()
        title = lines[0] if lines else f"Document {i + 1}"
        
        # Use the document ID or filename to reference an image (assuming you have matching images)
        image_filename = f"doc{i + 1}.jpg"  # Example: doc1.jpg, doc2.jpg, etc.
        image_path = os.path.join(document_image_folder, image_filename)
        
        # Extract a short description (first 200 characters as an example)
        description = content[:200]
        
        # Store metadata
        document_metadata.append({
            "title": title,
            "image": image_path,
            "description": description,
            "original_link": original_webpage_links[i]
        })

# Tokenization function
def tokenize(text):
    return text.lower().split()

# Term Frequency (TF)
def term_frequency(term, document):
    return document.count(term) / len(document)

# Inverse Document Frequency (IDF)
def inverse_document_frequency(term, all_documents):
    num_docs_containing_term = sum(1 for doc in all_documents if term in doc)
    return math.log(len(all_documents) / (1 + num_docs_containing_term))

# Compute TF-IDF
def compute_tfidf(document, all_documents, vocab):
    tfidf_vector = []
    for term in vocab:
        tf = term_frequency(term, document)
        idf = inverse_document_frequency(term, all_documents)
        tfidf_vector.append(tf * idf)
    return np.array(tfidf_vector)

# Cosine Similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Tokenize documents
tokenized_docs = [tokenize(doc) for doc in docs]

# Vocabulary creation
vocab = set([word for doc in tokenized_docs for word in doc])
vocab = sorted(vocab)

# Calculate TF-IDF vectors for documents
doc_tfidf_vectors = [compute_tfidf(doc, tokenized_docs, vocab) for doc in tokenized_docs]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    if not query:
        return render_template('index.html', error="Please enter a search query.")

    # Process the query
    query_vector = compute_tfidf(tokenize(query), tokenized_docs, vocab)

    # Calculate cosine similarities
    similarities = [cosine_similarity(query_vector, doc_vector) for doc_vector in doc_tfidf_vectors]
    sorted_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    # Prepare results with metadata
    results = []
    for i, sim in sorted_similarities:
        if sim > 0:  # Only show documents with similarity greater than 0
            doc_metadata = document_metadata[i]  # Get the corresponding metadata
            result = {
                "doc_id": i + 1,
                "score": sim,
                "doc": docs[i][:200],  # Short snippet
                "title": doc_metadata["title"],
                "image_filename": f"doc_{i + 1}.jpg",   
                "description": doc_metadata["description"],
                "original_link": doc_metadata["original_link"]
            }
            results.append(result)

    return render_template('index.html', query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)
