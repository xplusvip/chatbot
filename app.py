from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk
import os
import logging
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# Download NLTK data
nltk.download('punkt')

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configuration
CSV_FILE_PATH = 'Q_A_1445.csv'  # Set the correct path to your CSV file
CACHE_FILE_PATH = 'embeddingss_cache.json'

# Set up logging
logging.basicConfig(
    filename='logg.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

# Function to preprocess and vectorize questions
def preprocess_and_vectorize(df):
    logger.info("Preprocessing data and calculating TF-IDF vectors for all questions")
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    vectors = vectorizer.fit_transform(df['question'])
    logger.info("TF-IDF vectors calculated successfully")
    return vectorizer, vectors

# Function to get most similar answer to a given question and its similarity score using Cosine Similarity
def get_most_similar_answer(question, vectorizer, vectors, df):
    logger.info(f"Finding most similar answer for the question: {question}")
    question_vector = vectorizer.transform([question])
    similarity_scores = cosine_similarity(question_vector, vectors)
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
        
        loader = CSVLoader(file_path=file_path, encoding='utf-8')
        documents = loader.load()
        vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(api_key=api_key),
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return chain
    except Exception as e:
        logger.error(f"Error initializing chain: {e}")
        raise

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/question', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question', '')
        if not question:
            return jsonify({"error": "Question is required"}), 400

        df = load_data(CSV_FILE_PATH)
        vectorizer, vectors = preprocess_and_vectorize(df)
        
        answer, similarity_score = get_most_similar_answer(question, vectorizer, vectors, df)
        if similarity_score < 0.7:  # Check if similarity score is less than 70%
            chain = initialize_chain(CSV_FILE_PATH)
            result = chain({"question": question, "chat_history": []})
            answer = result['answer']

        return jsonify({"answer": answer, "similarity_score": similarity_score}), 200
    except Exception as e:
        logger.error(f"Error during question processing: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 1000))
    app.run(host='0.0.0.0', port=port)