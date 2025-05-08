import os
import pandas as pd
import streamlit as st
import uuid
import concurrent.futures
import difflib
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from llama_cpp import Llama

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#model_path = "Llama-3.2-1B-Instruct-IQ3_M.gguf"
model_path = "mistral-7b-instruct-v0.2.Q4_0.gguf"

st.set_page_config(
    page_title="Campus Connect AI",
    page_icon="CC",
    layout="wide"
)

file_names = [
    "study_permit_general", "work_permit_student_general", "work-study-data-llm",
    "vancouver_transit_qa_pairs", "permanent_residence_student_general", "data-with-sources",
    "faq_qa_pairs_general", "hikes_qa", "sfu-faq-with-sources", "sfu-housing-with-sources",
    "sfu-immigration-faq", "park_qa_pairs-up", "cultural_space_qa_pairs_up",
    "qa_pairs_food", "qa_pairs_year_and_month_avg", "qa_pairs_sfu_clubs"
]

collection_map = {
    "study_permit": "study_permit_general",
    "work_permit": "work_permit_student_general",
    "work_study": "work-study-data-llm",
    "public_transit": "vancouver_transit_qa_pairs",
    "permanent_residence": "permanent_residence_student_general",
    "health_related": "data-with-sources",
    "general_faqs": "faq_qa_pairs_general",
    "vancouver_hiking": "hikes_qa",
    "university_general_faqs": "sfu-faq-with-sources",
    "university_housing": "sfu-housing-with-sources",
    "university_immigration_faqs": "sfu-immigration-faq",
    "vancouver_parks": "park_qa_pairs-up",
    "vancouver_cultural": "cultural_space_qa_pairs_up",
    "university_food": "qa_pairs_food",
    "expenditure": "qa_pairs_year_and_month_avg",
    "university_clubs": "qa_pairs_sfu_clubs"
}

valid_categories = list(collection_map.keys())
fallback_category = "general_faqs"

# how did i fix the issue? am using caching now to prevent it from re loading data or re embedding documents, etc. 
# i only tested on llama + paraphrase, please test on mistral + all-miniLM 

@st.cache_resource(show_spinner="Loading model...")
def get_llama_model():
    return Llama(model_path=model_path, n_ctx=2048, n_threads=8, verbose=False)

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        #model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        model_kwargs={"use_auth_token": hf_token}
    )

@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

@st.cache_resource(show_spinner="Processing and embedding documents...")
def load_and_embed_documents(file_names, _client, _embedding_model, batch_size=32):
    print("embedding now...")
    collections = {}
    for file in file_names:
        path = f'../Data/{file}.csv'
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=lambda col: col.lower() in {"question", "answer"})
        df.columns = df.columns.str.lower()
        if "question" not in df.columns or "answer" not in df.columns:
            continue
        df = df.drop_duplicates(subset="question")
        df["text"] = df["question"].fillna('') + ' ' + df["answer"].fillna('')
        unique_texts = list(set(df["text"].dropna().tolist()))
        collection = _client.get_or_create_collection(name=file)
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i:i + batch_size]
            embeddings = embedding_model.embed_documents(batch)
            ids = [str(uuid.uuid4()) for _ in batch]
            collection.add(ids=ids, embeddings=embeddings, documents=batch)
        collections[file] = collection
    return collections

llama_model = get_llama_model()
embedding_model = get_embedding_model()
chroma_client = get_chroma_client()
collections = load_and_embed_documents(file_names, chroma_client, embedding_model)

def classify_query(query):
    category_prompt = f"""
    You are a classifier for a Q&A system for international students in British Columbia.
    Pick ONLY from this list of category names (copy them exactly, case-insensitive), and return up to 3 relevant ones (comma-separated):

    {", ".join(valid_categories)}

    Query: "{query}"

    Return only the category name(s) as a comma-separated string.
    """
    response = llama_model(category_prompt, max_tokens=50, temperature=0.1)["choices"][0]["text"].strip().lower()
    tokens = [t.strip() for t in response.split(",")]
    matched = []
    for token in tokens:
        if token in valid_categories and token not in matched:
            matched.append(token)
            continue
        closest = difflib.get_close_matches(token, valid_categories, n=1, cutoff=0.7)
        if closest and closest[0] not in matched:
            matched.append(closest[0])
        if len(matched) == 3:
            break
    if not matched:
        for category in valid_categories:
            if category in response and category not in matched:
                matched.append(category)
                if len(matched) == 3:
                    break
    if not matched:
        matched = [fallback_category]
    elif fallback_category not in matched and len(matched) < 3:
        matched.append(fallback_category)
    return matched[:3]

def get_relevant_documents(query, categories, n_results=3):
    print("getting relevant docs..")
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
    if len(categories) > 1 and len(all_results) > n_results:
        diverse_results = []
        seen_categories = set()
        for doc, sim in all_results:
            doc_category = next((cat for cat in categories if collection_map.get(cat) in doc), None)
            if doc_category and doc_category not in seen_categories:
                diverse_results.append((doc, sim))
                seen_categories.add(doc_category)
                if len(diverse_results) >= min(n_results, len(categories)):
                    break
        remaining_slots = n_results - len(diverse_results)
        if remaining_slots > 0:
            for doc, sim in all_results:
                if (doc, sim) not in diverse_results:
                    diverse_results.append((doc, sim))
                    if len(diverse_results) >= n_results:
                        break
        return diverse_results
    else:
        return all_results[:n_results]

def generate_answer(query):
    categories = classify_query(query)
    relevant_documents = get_relevant_documents(query, categories)
    print("relevant docs;")
    print(relevant_documents)
    if not relevant_documents:
        return {"Response": "Sorry, no relevant documents found."}
    seen = set()
    unique_docs = []
    for doc, sim in relevant_documents:
        doc_text = doc.split(" [Source: ")[0] if " [Source: " in doc else doc
        if doc_text not in seen:
            seen.add(doc_text)
            unique_docs.append((doc, sim))
    relevant_texts = "\n\n".join([doc.split(" [Source: ")[0] for doc, _ in unique_docs])

    # category-wise prompts
    hike_prompt = f"""
        INSTRUCTIONS:
            1. Convert structured information about the hike into a short, friendly paragraph using natural language. Do not repeat numbers or use formatting from the source.
            2. If they ask about hiking information, only answer with required information. Users can ask for more information if needed.
            3. When asked for a particular type of hike, find it instead of saying that one would not work in the category they asked for.
            4. Do NOT list trail attributes or stats (like “Distance: 3.1 km, Elevation: 789 m”). Instead, describe them in context (e.g., “a steep 3 km trail with a tough 789 m climb”).
            5. Avoid repeating exact numbers unless essential (e.g., elevation gain is helpful, but don't dump all stats).
    """
    
    parks_prompt = f""" 
        INSTRUCTIONS:
            1. Convert structured information about the park into a short, friendly paragraph using natural language. Do not repeat numbers or use formatting from the source.
            2. Provide only necessary information that will allow the user to enjoy the park.
                - Feel free to tell them about logisitical information if asked.
    """
    
    food_prompt = f""" 
        INSTRUCTIONS:
            1. Convert structured food and dining information into a friendly, helpful paragraph. Do not copy the question or use list formatting.
            2. Only answer what the user asked. Do NOT add information that wasn't requested.
            3. Describe details in a natural way (e.g., “open 24/7 during the semester” instead of “Hours: 24/7”).
            4. Mention unique features only when they help clarify the user's question.
            5. If a specific venue or program is asked about (e.g., a café, meal plan, or food station), describe it clearly in context.
            6. If the question can't be answered from the data, respond with: “I'm sorry, I don't have that information. Please check the official SFU Food website.”
            7. Provide the official link when available and relevant to the answer.
            8. Do NOT list menu items, prices, or square footage unless directly relevant to the user's question.
            9. Only provide food information that is relevant. If they ask for some place that serves a chicken sandwich do not provide information to a vegan place.
    """

    # activities general: covers how to answer general parks, hikes, food, clubs, cultural related questions 
    activities_general = f""" 
        INSTRUCTIONS:
            1. If they ask for suggestions, provide 2 to 3 suggestions.
            2. Do NOT list all information. Instead describe them in context 
            3. Provide accuracte suggestions, NOT suggestions of things that will not work for what they want.
            4. Convert structured information about the activity into a short, friendly paragraph using natural language. Do not repeat formatting from the source.
    """
    
    # permits prompt: covers ways to answer immigration, study permits, work permits, and permanent residence related questions 
    permits_prompt = f"""
        INSTRUCTIONS:
            1. When given a specific question with many possible answers, you can ask for more specific information.
                - if they are not asking for an extension do not provide information in regards to an extension of a permit.
            2. Only answer with information provided 
                - Information should NOT be guessed and do NOT add extra information
            3. If the answer is not in the dataset, respond with: "I'm sorry, I don't have that information. Please check the official IRCC website for more details."
            4. If it is helpful, provide the link and a description about it.
            5. Do NOT list all information. Instead describe them in context 
            6. If the answer depends on a specific condition explain those clearly.
            7. Do NOT make assumptions about the user's situation. 
    """
    
    housing_prompt = f""" 
        INSTRUCTIONS:
            1. Convert structured information about SFU or general student housing into a short, friendly paragraph using natural language. Do not repeat formatting or list prices unless helpful for context.
            2. Focus on what matters to the student: location, room types, meal plans, how to apply, and support available.
            3. Only mention costs in a general way (e.g., "starts around $4,000 per term") unless the user explicitly asks for detailed pricing.
            4. If information varies (e.g., by room type or campus), explain this clearly but briefly.
            5. If the user asks a specific housing question and the answer depends on certain conditions (e.g., term length, student status), explain those conditions clearly and simply.
            6. If the answer is not known or not in the data, respond with: "I’m sorry, I don’t have that information. Please check the SFU Housing website for details."
            7. Do NOT dump full lists of buildings, prices, or amenities. Summarize and keep it conversational.
            8. If the information is specific to SFU, make sure you say it to be clear.
    """
    
    transit_prompt = f""" 
        INSTRUCTIONS:
            1. Convert structured information about public transit into a short, friendly paragraph using natural language.
            2. Do NOT list statistics or technical formatting (like route numbers or fare charts) unless directly relevant to the user's question.
            3. Summarize relevant transit options clearly — describe them in context (e.g., “a quick SkyTrain ride from downtown to the airport”).
            4. Provide only what the user needs to understand how to get around or plan their trip.
            5. If the user is asking for directions, give a general summary of how they might travel.
            6. If the question is about fares, schedules, or route planning and the exact info is not available, tell the user to check the TransLink website and briefly explain what they can find there.
            7. Do NOT guess or make up transit information.
            8. If the information is not in the source, say “I’m sorry, I don’t have that information. You can check the official TransLink site for more details.”
    """
    
    # main rag prompt
    rag_prompt = f"""
    You are a helpful, friendly assistant for international students new to British Columbia, Canada.

    Below are some reference documents that may be relevant to the user's question:
    {relevant_texts}

    INSTRUCTIONS:
    1. If the user's query is just a greeting (like "hello", "hi", "what's up"):
       - Respond with a single brief friendly greeting
       - Offer to help with questions about studying or living in BC
       - Do NOT include ANY information from the reference documents
       - Do NOT create additional answers beyond answering their original question

    2. If the user is asking for information:
       - Be friendly and answer based ONLY on the reference documents if relevant
       - Summarize the necessary information into a couple sentences.
       - Do NOT create additional questions and answers beyond answering their original question
       - Limit your entire response to no more than 3 concise sentences when possible. Do not create long multi-line answers.
       - If the documents don't provide sufficient information, say "I don't have enough information to answer that. Please refer to official sources."
       - Ask for more information when there are multiple scenarios in the documents.
       - If they ask things like "can I", "will I", "how can I" feel free to ask follow up questions if you don't how to answer with the information provided. Do not just assume.
    
    3. IMPORTANT: Never generate additional content beyond answering the user's question. Do NOT number or bullet your points. Always use natural sentences and group similar information together where possible.
    
    User question: {query}

    Your response (just the answer, no preamble):
    """
    # adding the category specific prompting to main if necessary
    for category in categories:
        if category == "hiking" or category == "parks" or category == "food" or category == "cultural" or category == "clubs":
            rag_prompt += "\n" + activities_general
            
        if category == "hiking":
            rag_prompt += "\n" + hike_prompt
            
        if category == "parks":
            rag_prompt += "\n" + parks_prompt
        
        if category == "food":
            rag_prompt += "\n" + food_prompt
            
        if category == "study permit" or category == "work permit" or category == "immigration" or category == "permanent residence":
            rag_prompt += "\n" + permits_prompt
            
        if category == "housing":
            rag_prompt += "\n" + housing_prompt
        
        if category == "transit":
            rag_prompt += "\n" + transit_prompt
       
    # Add category-specific prompts if needed (not shown here)
    response_after_rag = llama_model(rag_prompt, max_tokens=300, temperature=0.1)["choices"][0]["text"]
    return {"Response": response_after_rag}

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
if "awaiting_bot_response" not in st.session_state:
    st.session_state.awaiting_bot_response = False

chat_container = st.container()

for message in st.session_state.messages:
    if message["role"] == "user":
        chat_container.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        chat_container.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)

if not st.session_state.awaiting_bot_response:
    if prompt := st.chat_input("Type a message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.awaiting_bot_response = True
        st.rerun()

else:
    with st.spinner("Campus Connect AI is thinking..."):
        prompt = st.session_state.messages[-1]["content"]  # last user message
        responses = generate_answer(prompt)
        bot_response = responses["Response"]
        st.session_state.messages.append({"role": "bot", "content": bot_response})
    st.session_state.awaiting_bot_response = False
    st.rerun()
