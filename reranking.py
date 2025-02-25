import streamlit as st
import google.generativeai as genai
import numpy as np
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from sentence_transformers import CrossEncoder



# Load environment variables from .env file
load_dotenv()

os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER"] = "false"

gemini_key = os.getenv("GEMINI_API_KEY")

# Configure the Google Generative AI model
genai.configure(api_key=gemini_key)

# Set up embedding functions
embedding_function = SentenceTransformerEmbeddingFunction()

# Initialize PDF reader
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]
pdf_texts = [text for text in pdf_texts if text]

# Set up text splitting
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# Set up ChromaDB

chroma_dir = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(chroma_dir, exist_ok=True)  # <-- This fixes the error

# 2. Initialize client with the directory
embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path=chroma_dir)
chroma_collection = chroma_client.get_or_create_collection(
    name="microsoft-collect", 
    embedding_function=embedding_function
)

# Add text chunks to Chroma
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)

# Function to create RAG prompt
def make_rag_prompt(query):
    prompt = ("""
        You are a knowledgeable financial research assistant. 
        Your users are inquiring about an annual report. 
        For the given question, propose up to five related questions to assist them in finding the information they need. 
        Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic. 
        Ensure each question is complete and directly related to the original inquiry. 
        List each question on a separate line without numbering.
    
        QUESTION: '{query}'
    """).format(query=query)
    return prompt

# Function to generate context from the query
def generate_context(original_query):
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = make_rag_prompt(original_query)
    answer = model.generate_content(prompt)
    answer_text = answer.text
    aug_questions = answer_text.split("\n")

    # concatenate the original query with the generated queries
    joint_queries = [original_query] + aug_questions

    results = chroma_collection.query(
        query_texts=joint_queries, n_results=10, include=["documents", "embeddings"]
    )

    retrieved_documents = results["documents"]

    # Deduplicate the retrieved documents
    unique_documents = set()
    for documents in retrieved_documents:
        for document in documents:
            unique_documents.add(document)

    unique_documents = list(unique_documents)

    pairs = []
    for doc in unique_documents:
        pairs.append([original_query, doc])

    # Cross-encoder for re-ranking the documents
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = cross_encoder.predict(pairs)

    top_indices = np.argsort(scores)[::-1][:5]
    top_documents = [unique_documents[i] for i in top_indices]

    # Concatenate the top documents into a single context
    context = "\n\n".join(top_documents)
    return context

# Function to generate final answer
def generate_final_prompt(query, context):
    prompt = ("""
    You are a professional financial research assistant with expertise in analyzing and summarizing annual reports.
    A user is inquiring about a specific aspect of an annual report, and you are expected to provide a well-structured and informative response.

    Based on the following context, which is a detailed extract from the annual report, please answer the query below:

    CONTEXT:
    {context}

    QUESTION:
    {query}

    Please ensure that your answer is clear, concise, and includes relevant details. Provide an overview that highlights the most important factors and trends.

    ANSWER:
    """).format(query=query, context=context)

    return prompt

# Function to generate prediction (this function will run in a separate thread)
def make_prediction(original_query):
    context = generate_context(original_query)
    prompt = generate_final_prompt(original_query, context)
    model = genai.GenerativeModel('gemini-2.0-flash')
    answer1 = model.generate_content(prompt)
    return answer1.text



# Streamlit interface
st.title("Microsoft Annual Report ")
user_input = st.text_input("Enter your question:")

if st.button("Generate"):
    if user_input:
        try:
            answer = make_prediction(user_input)
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    else:
        st.warning("Please enter a question.")