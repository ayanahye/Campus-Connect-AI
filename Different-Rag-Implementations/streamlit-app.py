import os
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import uuid
from llama_cpp import Llama

model_path = "/Users/poorvibhatia/Desktop/Projects/Campus-Connect-AI/Different-Rag-Implementations/Llama-3.2-1B-Instruct-IQ3_M.gguf"
model = Llama(model_path=model_path, n_ctx=2048, n_threads=8)
repo_id = "google/gemma-2-2b-it"
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
st._is_running_with_streamlit = True

file_names = [
    "General-Study-Permit-Questions/study_permit_general",
    "WorkPermits-For-Students/work_permit_student_general",
    "Work-Study-Data-and-Scripts/work-study-data-llm",
    "Transit-Data-Ques-Ans/vancouver_transit_qa_pairs",
    "Permanent-residence-for-students/permanent_residence_student_general",
    "Health-Data-and-Scripts-for-Chatbot/data-with-sources",
    "SFU-Faq-Data/sfu-faq-with-sources"
]

# all_texts = []

# for file in file_names:
#     path = f'/Users/poorvibhatia/Desktop/Projects/Campus-Connect-AI/{file}.csv'
#     try:
#         df = pd.read_csv(path)
#         df.columns = df.columns.str.lower()

#         if 'question' in df.columns and 'answer' in df.columns:
#             df = df.drop_duplicates(subset=['question'])
#             df['text'] = df['question'].fillna('') + ' ' + df['answer'].fillna('')
#         elif 'Question' in df.columns and 'Answer' in df.columns:
#             df = df.drop_duplicates(subset=['question'])
#             df['text'] = df['Question'].fillna('') + ' ' + df['Answer'].fillna('')
#         else:
#             print(f"no text columns in {file}")
#             continue
#         all_texts.extend(df['text'].tolist())
#     except Exception as e:
#         print(f"Error loading {file}: {e}")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="combined_docs")

# data = pd.DataFrame({"text": all_texts})
# data = data.drop_duplicates()
# all_texts = data["text"].tolist()

# print(f"successfully added {len(all_texts)} documents.")

def add_data_to_collection_batch(collection, texts, batch_size=3):
    for idx in range(0, len(texts), batch_size):
        try:
            batch_texts = texts[idx: idx + batch_size]

            embeddings = embedding_model.embed_documents(batch_texts)

            batch_ids = [str(uuid.uuid4()) for _ in batch_texts]

            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts
            )
            print(f"successfully added {len(batch_texts)} documents (Batch {idx}-{idx + batch_size - 1})")
        except Exception as e:
            print(f"Error processing batch starting at index {idx}: {e}")

# add_data_to_collection_batch(collection, all_texts)
# print(f"successfully added {len(all_texts)} documents to the Chroma collection.")

def get_relevant_documents(query, n_results=3):
    try:
        query_embeddings = embedding_model.embed_documents([query])[0]

        results = collection.query(query_embeddings=[query_embeddings], n_results=n_results)
        print(f"Query Results: {results}")

        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        print(f"Error querying: {e}")
        return []
    
def generate_answer(query):
    response_before_rag = model(query, max_tokens=200, temperature=0.1)["choices"][0]["text"]

    relevant_documents = get_relevant_documents(query)
    if not relevant_documents:
        return {
            "Before RAG Response": response_before_rag,
            "After RAG Response": "Sorry, no relevant documents found."
        }

    relevant_texts = "\n\n".join(relevant_documents)
    rag_prompt = f"""
    You are a helpful assistant for international students. Here are relevant documents:

    {relevant_texts}

    Please respond to the following question based on the documents above. Be conversational but concise:

    Question: {query}

    Answer:
    """

    response_after_rag = model(rag_prompt, max_tokens=300, temperature=0.1)["choices"][0]["text"]

    return {
        "Before RAG Response": response_before_rag,
        "After RAG Response": response_after_rag
    }

import nest_asyncio
nest_asyncio.apply()

st.title("Simple Chatbot")

user_query = st.text_input("Ask me a question: ")

if st.button("Generate Answer"):
    if user_query:
        responses = generate_answer(user_query)
        print(responses["After RAG Response"])
        st.write(responses["After RAG Response"])
    else:
        st.warning("Please enter a query.")

# user_query = "How do I commute in vancouver and how can I get to SFU?"
# responses = generate_answer(user_query, category)

# print("User Query:", user_query)
# print("Response Before RAG:", responses["Before RAG Response"])
# print("Response After RAG:", responses["After RAG Response"])