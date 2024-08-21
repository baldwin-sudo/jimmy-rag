import os
import replicate
from langchain.schema import Document
from langchain_community.vectorstores import ElasticVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from elasticsearch import Elasticsearch
from tqdm import tqdm
import time
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure you have NLTK data downloaded
nltk.download('punkt')


# Elasticsearch connection setup
es = Elasticsearch("http://localhost:9200")

# Load text files from the 'texts' directory
text_directory = 'texts'
text_files = [os.path.join(text_directory, file) for file in os.listdir(text_directory) if file.endswith('.txt')]

index_name = "jimmy_test"

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to process a single text file and return documents with metadata
def process_text_file(text_file):
    documents = []
    
    # Read the text from the file
    text = read_text_file(text_file)
    
    # For simplicity, we're considering each text file as a single document
    doc = Document(
        page_content=text,
        metadata={
            'source_file': text_file
        }
    )
    documents.append(doc)
    print(f"Added document for {text_file}")  # Debug print
    return documents

# Load and process text files
all_documents = []
for text_file in tqdm(text_files, desc="Processing Text Files"):
    all_documents.extend(process_text_file(text_file))

print(f"Total documents processed: {len(all_documents)}")  # Debug print

# Function to chunk text using NLTK
def chunk_text_nltk(text, max_chunk_size=2000):  # Reduced max_chunk_size to increase the number of chunks
    chunks = []
    sentences = sent_tokenize(text)  # Tokenize text into sentences

    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(word_tokenize(sentence))
        if current_size + sentence_size <= max_chunk_size:
            current_chunk.append(sentence)
            current_size += sentence_size
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Split documents into smaller chunks
document_chunks = []
for doc in tqdm(all_documents, desc="Chunking documents"):
    chunks = chunk_text_nltk(doc.page_content)
    for chunk in chunks:
        document_chunks.append(Document(page_content=chunk, metadata=doc.metadata))

print(f"Total chunked documents: {len(document_chunks)}")  # Debug print

# Use HuggingFace embeddings with 'all-mpnet-base-v2' model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/bert-large-nli-stsb-mean-tokens")

# Profile the embedding and storage process
start_time = time.time()
vector_store = ElasticVectorSearch.from_documents(document_chunks, embeddings, elasticsearch_url="http://localhost:9200", index_name=index_name)
end_time = time.time()
print(f"Time taken to embed and store documents: {end_time - start_time} seconds")

# Define the Query Function
class CustomConversationalRetrievalChain:
    def __init__(self, vector_store, embeddings):
        self.vector_store = vector_store
        self.embeddings = embeddings

    def get_relevant_documents(self, prompt):
        docs = self.vector_store.similarity_search(prompt, k=10)  # Increase k to retrieve more relevant documents
        if not docs:
            print("No documents found for the given query.")
        else:
            for i, doc in enumerate(docs):
                print(f"Document {i+1}:")
                print(f"Page Content: {doc.page_content[:500]}...")  # Print the first 500 characters of the document
                print(f"Metadata: {doc.metadata}")  # Print metadata with source and page
        return docs

    def query(self, prompt, chat_history):
        docs = self.get_relevant_documents(prompt)[:]
        if not docs:
            return {"answer": "No relevant documents found.", "source_documents": []}
        combined_text = "\n".join([doc.page_content for doc in docs])
        combined_text = combined_text
     
        combined_text += f"\n\n based on the provided documents answer the following Question: {prompt}"  # Add the question at the end for focus
     
        input_data = {
            "prompt": combined_text,
            "temperature": 0.7,
            "max_new_tokens": 150
        }

        result = client.run("mistralai/mistral-7b-v0.1", input=input_data)
        answer = "".join(result)
        print("Raw Model Output:", answer)  # Debug print
        return {"answer": answer, "source_documents": [doc.metadata for doc in docs]}  # Return metadata

# Test the query with a sample question
qa_chain = CustomConversationalRetrievalChain(vector_store, embeddings)
test_question = "What are the copayment details and benefits for advanced level hearing aids, and what is included in the purchase?"
result = qa_chain.query(test_question, chat_history=[])
print('****************Answer:', result['answer'])
