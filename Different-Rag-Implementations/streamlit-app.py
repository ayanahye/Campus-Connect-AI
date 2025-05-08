import os
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import streamlit as st
import os
import chromadb
import uuid
from llama_cpp import Llama
import re
import difflib
import concurrent.futures
import chromadb


model_path = "mistral-7b-instruct-v0.2.Q4_0.gguf"
model = Llama(model_path=model_path, n_ctx=2048, n_threads=8, verbose=False)
st._is_running_with_streamlit = True

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# persistent client to interact with chroma vector store
client = chromadb.PersistentClient(path="./chroma_db")

# create collections for each data
collection = client.get_or_create_collection(name="combined_docs")


prefix = "../all_data/"

file_names = [
    "study_permit_general", "work_permit_student_general", "work-study-data-llm",
    "vancouver_transit_qa_pairs", "permanent_residence_student_general", "data-with-sources",
    "faq_qa_pairs_general", "hikes_qa", "sfu-faq-with-sources", "sfu-housing-with-sources",
    "sfu-immigration-faq", "park_qa_pairs-up", "cultural_space_qa_pairs_up",
    "qa_pairs_food", "qa_pairs_year_and_month_avg", "qa_pairs_sfu_clubs"
]

collection_map = {
    "study": "study_permit_general",
    "student work": "work_permit_student_general",
    "work-study": "work-study-data-llm",
    "transit": "vancouver_transit_qa_pairs",
    "permanent residence": "permanent_residence_student_general",
    "general info": "data-with-sources",
    "faq": "faq_qa_pairs_general",
    "hiking": "hikes_qa",
    "sfu faq": "sfu-faq-with-sources",
    "housing": "sfu-housing-with-sources",
    "immigration": "sfu-immigration-faq",
    "parks": "park_qa_pairs-up",
    "culture": "cultural_space_qa_pairs_up",
    "food": "qa_pairs_food",
    "weather": "qa_pairs_year_and_month_avg",
    "clubs": "qa_pairs_sfu_clubs"
}

#file_names = [prefix + file for file in file_names]

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
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    model_kwargs={"use_auth_token": hf_token}
)

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="combined_docs")

# data = pd.DataFrame({"text": all_texts})
# data = data.drop_duplicates()
# all_texts = data["text"].tolist()

# print(f"successfully added {len(all_texts)} documents.")

collections = {}
batch_size = 32


def process_file(file):
    try:
        path = prefix + file + ".csv"
        #path = f'../Data/{file}.csv'
        if not os.path.exists(path):
            return f"{file} skipped (file not found)."

        df = pd.read_csv(path, usecols=lambda col: col.lower() in {"question", "answer"})
        df.columns = df.columns.str.lower()

        if "question" not in df.columns or "answer" not in df.columns:
            return f"{file} skipped (missing question/answer columns)."

        df = df.drop_duplicates(subset="question")
        df["text"] = df["question"].fillna('') + ' ' + df["answer"].fillna('')
        unique_texts = list(set(df["text"].dropna().tolist()))

        collection = client.get_or_create_collection(name=file)
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i:i + batch_size]
            embeddings = embedding_model.embed_documents(batch)
            ids = [str(uuid.uuid4()) for _ in batch]
            collection.add(ids=ids, embeddings=embeddings, documents=batch)

        collections[file] = collection
        return f"{file}: Loaded {len(unique_texts)} docs."
    except Exception as e:
        return f"{file}: Error - {e}"

# parallelogram
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    results = list(executor.map(process_file, file_names))

for result in results:
    print(result)

# add_data_to_collection_batch(collection, all_texts)
# print(f"successfully added {len(all_texts)} documents to the Chroma collection.")

def get_relevant_documents(query, categories, n_results=2):
    all_results = []
    query_embedding = embedding_model.embed_documents([query])[0]

    for category in categories:
        collection_name = collection_map[category]
        if collection_name in collections:
            try:
                result = collections[collection_name].query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                docs = result.get("documents", [[]])[0]
                sims = result.get("distances", [[]])[0]

                all_results.extend(zip(docs, sims))
            except Exception as e:
                print(f"error querying {collection_name}: {e}")

    all_results = sorted(all_results, key=lambda x: x[1])

    return all_results[:n_results]


valid_categories = list(collection_map.keys())
fallback_category = "faq"

def classify_query(query):
    category_prompt = f"""
    You are a classifier for a Q&A system for international students in British Columbia.
    Choose the **1 most relevant** category from this list, or at most 3 if absolutely needed (comma-separated):

    {", ".join(valid_categories)}

    Query: "{query}"

    Return only the category name(s) as a comma-separated string.
    """

    response = model(category_prompt, max_tokens=50, temperature=0)["choices"][0]["text"].strip().lower()
    print("Raw out:", response)

    tokens = re.findall(r'\b\w+\b', response)

    matched = []
    for token in tokens:
        closest = difflib.get_close_matches(token, valid_categories, n=1, cutoff=0.8)
        if closest and closest[0] not in matched:
            matched.append(closest[0])
        if len(matched) == 3:
            break

    if fallback_category not in matched:
      matched.append(fallback_category)

    return matched[:3]

    
def generate_answer(query):
    categories = classify_query(query)
    print(f"Categories {categories}\n")
    relevant_documents = get_relevant_documents(query, categories)

    if not relevant_documents:
        return {
            "After RAG Response": "Sorry, no relevant documents found."
        }

    #relevant_documents = list(set(relevant_documents))

    seen = set()
    unique_docs = []
    for doc, sim in relevant_documents:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append((doc, sim))

    print("Relevant Documents with Similarity Scores:")
    for doc, sim in unique_docs:
        print(f"Similarity: {sim:.4f}\nDoc: {doc}\n")

    relevant_texts = "\n\n".join([doc for doc, _ in unique_docs])

    rag_prompt = f"""
    You are a helpful assistant for international students new to British Columbia Canada. Here are relevant documents:

    {relevant_texts}

    Please respond to the following question. Be conversational but concise, aim to answer accurately using the documents, but in as few words as possible (i.e. less than 20). DO NOT USE THE DOCUMENTS IF THEY ARE NOT HELPFUL FOR THE QUERY. Do not ask the user irrelevant questions unless it relates to their query.
    Question: {query}

    Answer:
    """

    response_after_rag = model(rag_prompt, max_tokens=300, temperature=0.1)["choices"][0]["text"]

    return {
        "After RAG Response": response_after_rag
    }


import streamlit as st

st.set_page_config(page_title="Campus Connect AI", page_icon="CC", layout="wide")

st.markdown("""
     <style>
        @import url('https://fonts.googleapis.com/css2?family=Patrick+Hand&display=swap');

    .simple-header {
        position: sticky;
        top: 0;
        z-index: 9999;
        background: linear-gradient(90deg, #cce6ff 0%, #99ccff 100%);
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        padding: 0.5rem 0.5rem 0.25rem 0.5rem;
        border-radius: 0 0 8px 8px;
        font-family: 'Patrick Hand', cursive;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        user-select: none;
        margin-bottom: 20px;
    }

    .simple-header h1 {
        font-size: 3rem;
        margin: 0;
        padding: 0;
        color: #003366;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.15);
        font-family: 'Patrick Hand', cursive;
    }

    .simple-header .subheader {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.25rem;
        color: #0059b3;
        margin-top: 0.25rem;
        font-weight: 600;
    }

    .user-message {
        background-color: white !important;
        border-right: 4px solid black !important;
        margin-left: auto;
        margin-bottom: 1rem;
        border-radius: 18px;
        padding: 12px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        max-width: 70%;
        animation: fadeIn 0.3s ease-out;
    }

    .bot-message {
        background-color: white !important;
        border-left: 4px solid black !important;
        margin-right: auto;
        margin-bottom: 1rem;
        border-radius: 18px;
        padding: 12px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        max-width: 70%;
        animation: fadeIn 0.3s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    body, .reportview-container, .stApp {
        background: linear-gradient(to bottom, #cce6ff, white) !important;
        background-attachment: fixed !important;
        min-height: 100vh;
    }

    .stChatInput {
        border: 1px solid #d1d5db !important;
        border-radius: 999px !important;
        background: white !important;
        padding: 8px 12px !important;
    }

    .stChatInput:focus-within {
        box-shadow: 0 0 0 2px rgba(0,0,0,0.1) !important;
    }

    .stButton>button {
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        background: white !important;
        border: 1px solid #d1d5db !important;
    }

    @media (max-width: 768px) {
        .user-message, .bot-message {
            max-width: 85% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('''
    <div class="simple-header">
        <h1>Campus Connect AI</h1>
        <div class="subheader">Chatbot for International Students</div>
    </div>
''', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_container = st.container()

for message in st.session_state.messages:
    if message["role"] == "user":
        chat_container.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        chat_container.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Type a message here..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    responses = generate_answer(prompt) 
    bot_response = responses["After RAG Response"]
    st.session_state.messages.append({"role": "bot", "content": bot_response})
    
    st.rerun()

# user_query = "How do I commute in vancouver and how can I get to SFU?"
# responses = generate_answer(user_query, category)

# print("User Query:", user_query)
# print("Response Before RAG:", responses["Before RAG Response"])
# print("Response After RAG:", responses["After RAG Response"])

#running:
#- streamlit run streamlit-app.py
#or
#- python -m streamlit run streamlit-app.py

#http://localhost:8501

