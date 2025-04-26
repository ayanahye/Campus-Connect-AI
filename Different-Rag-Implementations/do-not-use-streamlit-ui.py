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
import re
import difflib
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#model_path = "/Users/poorvibhatia/Desktop/Projects/Campus-Connect-AI/Different-Rag-Implementations/Llama-3.2-1B-Instruct-IQ3_M.gguf"
model_path = "./Llama-3.2-1B-Instruct-IQ3_M.gguf"
model = Llama(model_path=model_path, n_ctx=2048, n_threads=8)
#repo_id = "google/gemma-2-2b-it"
#hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
st._is_running_with_streamlit = True

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
    '''
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
    '''
    rag_prompt = f"""
    You are a helpful assistant for international students new to British Columbia Canada. 

    Please respond to the following question. Be conversational but concise.
    Question: {query}

    Answer:
    """

    response_after_rag = model(rag_prompt, max_tokens=300, temperature=0.1)["choices"][0]["text"]

    return {
        "After RAG Response": response_after_rag
    }



import streamlit as st
import pandas as pd
import nest_asyncio
nest_asyncio.apply()

st.set_page_config(page_title="Campus Connect AI", page_icon="CC", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom, #cce6ff, white);
    }
    .reportview-container {
        background: linear-gradient(to bottom, #cce6ff, white);
    }
            
    .stApp {
        background: linear-gradient(to bottom, #cce6ff, white); !important;
        background-attachment: fixed !important;
        min-height: 100vh;
    }
            
    .header {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: black;
        
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }

    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
        max-width: 600px;  
        margin: 0 auto; 
    }

    .user-message {
        background-color: white;
        border: 1px solid grey;
        border-right: 4px solid black;
        padding: 8px;
        border-radius: 10px;
        max-width: 70%;
        margin-left: auto;  
        margin-bottom: 1rem;  
    }

    .bot-message {
        background-color: white;
        border: 1px solid grey;
        border-left: 4px solid black;
        padding: 8px;
        border-radius: 10px;
        max-width: 70%;
        margin-right: auto;  
        margin-top: 1rem;  
        margin-bottom: 1rem;
    }

    .input-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        width: 100%;
        background-color: white;
        border: 1px solid black;
        border-radius: 999px;
        padding: 0.25rem;
       
    }

    .text-input {
        flex: 1;
        padding: 0.75rem;
        border: none; 
        background-color: white;
        border-radius: 999px;
    }
            
    stHorizontalBlock {
      background-color: white;  
    }

    .send-button {
        padding: 0.75rem;
        background-color: white;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        border: none;  
        border-radius: 50%;
        background-color: white;
    }

    .send-button svg {
        width: 20px;
        height: 20px;
        fill: black;
    }
            
    .st-ba.st-bw.st-bx.st-by.st-bz.st-c0.st-c1.st-c2.st-c3.st-c4.st-c5.st-b8.st-c6.st-c7.st-c8.st-c9.st-ca.st-cb.st-cc.st-cd.st-ae.st-af.st-ag.st-ce.st-ai.st-aj.st-bv.st-cf.st-cg.st-ch {
        background-color: white !important;  
        border: 0px solid black !important;  
    }
            
    .st-emotion-cache-ocsh0s {
        border: none !important; 
        background: transparent !important;  
        padding: none !important;
        background: white !important; 
    }
            
     .stForm.st-emotion-cache-4uzi61.e1ttwmlf1,
    .stVerticalBlockBorderWrapper,
    .st-emotion-cache-0.eu6p4el5,
    .st-emotion-cache-b95f0i.eu6p4el4,
    .stVerticalBlock.st-emotion-cache-18b46wa.eu6p4el3 {
        padding: 1px !important;
    }
            
            
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">Campus Connect AI</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([11, 1])
        with col1:
            user_query = st.text_input("", placeholder="Type a message here...", key="user_input", label_visibility="collapsed")
        with col2:
            submit_button = st.form_submit_button("➔")
    st.markdown('</div>', unsafe_allow_html=True)

if submit_button and user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    responses = generate_answer(user_query) 
    bot_response = responses["After RAG Response"]
    st.session_state.messages.append({"role": "bot", "content": bot_response})
    st.rerun()


# user_query = "How do I commute in vancouver and how can I get to SFU?"
# responses = generate_answer(user_query, category)

# print("User Query:", user_query)
# print("Response Before RAG:", responses["Before RAG Response"])
# print("Response After RAG:", responses["After RAG Response"])


#running:
#- streamlit run do-not-use-streamlit-ui.py
#or
#- python -m streamlit run do-not-use-streamlit-ui.py

#http://localhost:8501


