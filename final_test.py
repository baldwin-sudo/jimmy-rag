import streamlit as st
import openai
import os
from langchain.vectorstores import ElasticVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from elasticsearch import Elasticsearch
from textwrap import wrap

# Set up OpenAI API key
def setup_environment():
    openai.api_key = 'sk-proj-G0OtjyLzDzWxF1W7PTb8T3BlbkFJfBsZnzuUEI7kvT1sQsbL'

# Initialize Elasticsearch
def initialize_elasticsearch():
    return Elasticsearch("http://localhost:9200")

# Initialize HuggingFace embeddings
def initialize_embeddings(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)

# Initialize vector store
def initialize_vector_store(es, index_name, embeddings):
    return ElasticVectorSearch(
        "http://localhost:9200",
        index_name,
        embeddings
    )

# Chunk the text into smaller segments
def chunk_text(text, chunk_size=4000, overlap=100):
    """
    Chunk text into smaller segments with optional overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Load all documents from /texts directory and chunk them
def load_documents_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Assuming text files
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                chunks = chunk_text(content)
                for i, chunk in enumerate(chunks):
                    documents.append({"filename": filename, "chunk_index": i, "content": chunk})
    return documents

# Load documents incrementally and update context
def load_and_update_context(vector_store, docs_list, message_history, model_name="gpt-3.5-turbo"):
    questions = []
    for doc in docs_list[:2]:
        # Add document to the context
        message_history.append({"role": "system", "content": doc['content']})
        
        # Query to generate questions based on current context
        prompt = "Based on the following document content, generate 5 potential questions a user might ask:\n\n" \
                 "Questions:"

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=message_history + [{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
            top_p=0.9
        )
        generated_questions = response['choices'][0]['message']['content'].strip()
        questions.append(generated_questions)
        
        # Update message history with generated questions
        message_history.append({"role": "assistant", "content": generated_questions})
        
    return questions, message_history

# Answer user queries based on similarity and embeddings
def answer_user_query(vector_store, user_query, message_history=None, model_name="gpt-3.5-turbo"):
    # Search for relevant documents based on similarity
    docs = vector_store.similarity_search(user_query, k=5)
    context = "\n\n---\n\n".join([f"Document - {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    full_prompt = f"{context}\n\nBased on the provided documents, answer the following question:\n{user_query}\n\n"

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=150,
        temperature=0.7,
        top_p=0.9
    )
    answer = response['choices'][0]['message']['content'].strip()
    return {"answer": answer, "source_documents": context}

# Main Streamlit app
def main():
    st.title("Document Query App")

    # Initialize components
    setup_environment()
    es = initialize_elasticsearch()
    
    embeddings = initialize_embeddings("sentence-transformers/bert-large-nli-stsb-mean-tokens")
    vector_store = initialize_vector_store(es, "jimmy_test", embeddings)
    
    # Initialize message history for context
    message_history = []

    # User input
    query_type = st.selectbox("Select Query Type", ["Generate Questions", "Answer Query"])
    
    if query_type == "Generate Questions":
        # Load documents from /texts directory
        if st.button("Load All Documents and Generate Questions"):
            docs_list = load_documents_from_directory("./texts")
            questions, message_history = load_and_update_context(vector_store, docs_list, message_history)
            
            # Display generated questions and message history
            st.write("**Generated Questions:**")
            for q in questions:
                st.write(q)
            
            
    elif query_type == "Answer Query":
        # User input for answering queries
        user_query = st.text_input("Enter your query:", "")
        if st.button("Get Answer"):
            if user_query:
                response = answer_user_query(vector_store, user_query, message_history)
                
                # Display the answer and source documents
                st.write("**Answer:**")
                st.write(response['answer'])
                
              
            else:
                st.write("Please enter a query.")

if __name__ == "__main__":
    main()
