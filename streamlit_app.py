import streamlit as st
import openai
from elasticsearch import Elasticsearch
from final_test import (
    setup_environment,
    initialize_elasticsearch,
    initialize_embeddings,
    initialize_vector_store,
    query_first_set,
    query_second_set,
    compare_answers,
)

def main():
    st.title("Document Query App")

    # Initialize components
    setup_environment()
    es = initialize_elasticsearch()
    
    # Initialize embeddings and vector stores
    embeddings_bert = initialize_embeddings("sentence-transformers/bert-large-nli-stsb-mean-tokens")
    vector_store_bert = initialize_vector_store(es, "jimmy_test", embeddings_bert)
    
    embeddings_mpnet = initialize_embeddings("sentence-transformers/all-mpnet-base-v2")
    vector_store_mpnet = initialize_vector_store(es, "test2", embeddings_mpnet)
    
    # Model selection
    model_name = st.selectbox(
        "Choose GPT Model:",
        ["gpt-4", "gpt-3.5-turbo"]
    )

    # User input
    user_query = st.text_input("Enter your query:", "")
    
    if st.button("Submit"):
        if user_query:
            # Query both sets
            result_bert = query_first_set(vector_store_bert, user_query, model_name)
             # Display results from BERT index
            st.write('**Answer from BERT:**', result_bert['answer'])
            st.write('**Source Documents from BERT:**')
            for doc in result_bert['source_documents']:
                st.write(doc.page_content)
            
            # Display results from MPNet index
            st.write('**Answer from MPNet:**', result_mpnet['answer'])
            st.write('**Source Documents from MPNet:**')
            for doc in result_mpnet['source_documents']:
                st.write(doc.page_content)
            
            # Compare answers
            comparison_result = compare_answers(result_bert, result_mpnet, user_query, model_name)
            st.write('**Best Answer After Comparison:**', comparison_result)
            
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()
