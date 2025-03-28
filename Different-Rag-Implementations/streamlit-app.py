from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import chromadb
import streamlit as st

repo_id = "google/gemma-2-2b-it"

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    task='text-generation',
    repo_id=repo_id,
    model="google/gemma-2-2b-it",
    max_length=1024,
    temperature=0.1,
    huggingfacehub_api_token=hf_token
)

health_data = pd.read_csv('./Health-Data-and-Scripts-for-Chatbot/data-with-sources.csv')
work_data = pd.read_csv('./Work-Study-Data-and-Scripts/work-and-education-data.csv')
transit_data = pd.read_csv('./Transit-Data-Ques-Ans/vancouver_transit_qa_pairs.csv')

health_data_sample = health_data
work_data_sample = work_data
transit_data_sample = transit_data

health_data_sample['text'] = health_data_sample['Question'].fillna('') + ' ' + health_data_sample['Answer'].fillna('')
work_data_sample['text'] = work_data_sample['Theme'].fillna('') + ' ' + work_data_sample['Content'].fillna('')
transit_data_sample['text'] = transit_data_sample['question'].fillna('') + ' ' + transit_data_sample['answer'].fillna('')

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="./chroma_db")

health_collection = client.get_or_create_collection(name="health_docs")
work_collection = client.get_or_create_collection(name="work_docs")
transit_collection = client.get_or_create_collection(name="transit_docs")

def add_data_to_collection(collection, data):
    for idx, row in data.iterrows():
        try:
            embeddings = embedding_model.embed_documents([row['text']])[0]
            collection.add(
                ids=[str(idx)],
                embeddings=[embeddings],
                documents=[row['text']]
            )
        except Exception as e:
            print(f"Error on index {idx}: {e}")

# add_data_to_collection(health_collection, health_data_sample)
# add_data_to_collection(work_collection, work_data_sample)
# add_data_to_collection(transit_collection, transit_data_sample)

def get_relevant_document(query, category):
    try:
        # get the embedding for the user query using same embedding model
        query_embeddings = embedding_model.embed_documents([query])[0]

        # choose the correct collection based on the category
        if category == "health":
            collection = health_collection
        elif category == "work":
            collection = work_collection
        elif category == "transit":
            collection = transit_collection

        # query the collection
        results = collection.query(query_embeddings=[query_embeddings], n_results=3)

        print(f"Query Results: {results}")
        return results['documents'][0][0] if results['documents'] else None
    except Exception as e:
        print(f"Error querying: {e}")
        return None
    
def generate_answer(query, category):
    output_before_rag = llm.predict(f"Respond to this question: {query}")
    response_before_rag = output_before_rag

    relevant_document = get_relevant_document(query, category)
    if relevant_document is None:
        return f"Sorry, no relevant document found. Model's response before RAG: {response_before_rag}"

    relevant_document = " ".join(relevant_document.split())
    MAX_DOC_LENGTH = 500
    relevant_document = relevant_document[:MAX_DOC_LENGTH]

    # rag_prompt = f"""
    # You are a helpful assistant for international students new to B.C. Here is a relevant document:

    # {relevant_document}

    # Please respond to the following question based on the document above:

    # Question: {query}

    # Answer:
    # """
    rag_prompt = f"""
    You are a helpful assistant for international students new to B.C. Here is a relevant document:

    {relevant_document}

    Please respond to the following question based on the document above, if you can't answer anything or it requires the international student to ask a query again, direct them to additional resources like the vancouver transit website or the transit mobile app for transit related queries:

    Question: {query}

    Answer:
    """

    output_after_rag = llm.predict(rag_prompt)
    response_after_rag = output_after_rag

    return {
        "Before RAG Response": response_before_rag,
        "After RAG Response": response_after_rag
    }

st.title("Simple Chatbot")

user_query = st.text_input("Ask me anything:")
category = "transit"

if st.button("Get Answer"):
    if user_query:
        responses = generate_answer(user_query, category)
        for response in responses:
            st.write(response)
    else:
        st.warning("Please enter a query.")

# user_query = "How do I commute in vancouver and how can I get to SFU?"
# responses = generate_answer(user_query, category)

# print("User Query:", user_query)
# print("Response Before RAG:", responses["Before RAG Response"])
# print("Response After RAG:", responses["After RAG Response"])