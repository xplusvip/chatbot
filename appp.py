import os
import pandas as pd
import torch
import logging
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from token import Tokenizer, EmbeddingModel  # Example import for token
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'your_secret_key'
CSV_FILE_PATH = 'Q_A_1445.csv'  # Set the correct path to your CSV file
CACHE_FILE_PATH = 'embeddingss_cache.json'

# Set up logging
logging.basicConfig(
    filename='logg.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    # handlers=[logging.FileHandler("logg.log"), logging.StreamHandler()]
)
logger = logging.getLogger()

# Load your CSV data into a DataFrame
def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        logger.info("Data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

# Load or initialize the cache
def load_cache():
    if os.path.exists(CACHE_FILE_PATH):
        with open(CACHE_FILE_PATH, 'r', encoding='utf-8') as cache_file:
            return json.load(cache_file)
    else:
        return {}

# Save the cache
def save_cache(cache):
    with open(CACHE_FILE_PATH, 'w', encoding='utf-8') as cache_file:
        json.dump(cache, cache_file)

# Initialize tokenizer and model (token library)
logger.info("Loading tokenizer and embedding model")
tokenizer = Tokenizer.load_pretrained('your-pretrained-tokenizer')  # Adjust this line based on token library
embedding_model = EmbeddingModel.load_pretrained('your-pretrained-embedding-model')  # Adjust this line based on token library

# Function to calculate embeddings using token library
def get_embeddings(text):
    logger.info(f"Calculating embeddings for: {text}")
    tokens = tokenizer.encode(text, max_length=512, truncation=True)
    embeddings = embedding_model.get_embeddings(tokens)
    return embeddings

# Function to preprocess and get embeddings for all questions
def preprocess_and_get_embeddings(df, cache):
    logger.info("Preprocessing data and calculating embeddings for all questions")
    embeddings = []
    for index, row in df.iterrows():
        question = row['question']
        if question in cache:
            embeddings.append(cache[question])
        else:
            embedding = get_embeddings(question).squeeze()
            embeddings.append(embedding)
            cache[question] = embedding.tolist()  # Convert numpy array to list for JSON serialization
    logger.info("Embeddings calculated successfully")
    save_cache(cache)  # Save the updated cache
    return embeddings

# Function to handle no reply scenario
def handle_no_reply():
    logger.info("Handling no reply scenario")
    return "لا يوجد رد"

# Function to get most similar answer to a given question and its similarity score using Cosine Similarity
def get_most_similar_answer(question, df, embeddings):
    logger.info(f"Finding most similar answer for the question: {question}")
    question_embedding = get_embeddings(question).squeeze()
    similarity_scores = cosine_similarity([question_embedding], embeddings)
    best_index = similarity_scores.argmax()  # For cosine similarity, we need the maximum value
    answer = df.iloc[best_index]['answer']
    similarity_score = similarity_scores[0][best_index]
    logger.info(f"Most similar answer found: {answer} with similarity score: {similarity_score}")
    return answer, similarity_score

# Function to initialize and use the Conversational Retrieval Chain
def initialize_chain(file_path):
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        os.environ["OPENAI_API_KEY"] = api_key  # Use environment variables for API key
        loader = CSVLoader(file_path=file_path, encoding='utf-8')
        documents = loader.load()
        vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
        chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(), retriever=vectorstore.as_retriever())
        return chain
    except Exception as e:
        logger.error(f"Error initializing chain: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/question', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question', '')
        if not question:
            raise ValueError("Question is required")

        df = load_data(CSV_FILE_PATH)
        cache = load_cache()
        embeddings = preprocess_and_get_embeddings(df, cache)
        
        answer, similarity_score = get_most_similar_answer(question, df, embeddings)
        if similarity_score < 0.81:  # Check if similarity score is less than 70%
            answer = "لا يوجد رد"
        
        if answer == "لا يوجد رد":
            try:
                chain = initialize_chain(CSV_FILE_PATH)
                result = chain.invoke({"question": question, "chat_history": []})
                answer = result['answer']
                return jsonify({"answer": answer, "similarity_score": similarity_score}), 200
            except Exception as e:
                logger.error(f"Error getting answer from chain: {e}")
                return jsonify({"error": handle_no_reply(), "similarity_score": similarity_score}), 500
        else:
            return jsonify({"answer": answer, "similarity_score": similarity_score}), 200
    except Exception as e:
        logger.error(f"Error during question processing: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()  # Changed to port 5001
