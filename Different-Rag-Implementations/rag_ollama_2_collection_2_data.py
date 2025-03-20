# example of using RAG with Ollama with 2 collections and 2 data files

'''
To use this script:
1. make sure ollama is installed: https://ollama.com/download
2. run `ollama pull phi:latest`
3. run `ollama pull mxbai-embed-large`
4. run the python file

dont worry too much this is a test and we can try hf 
'''

# note for distinct data im going to use another file for 2 separate collections. This is just a demo file if i had used 1 collection.
import pandas as pd
import ollama
import chromadb

# replace this with your own csv files
health_data = pd.read_csv('../Health-Data-and-Scripts-for-Chatbot/data-with-sources.csv')
work_data = pd.read_csv('../Work-Study-Data-and-Scripts/work-and-education-data.csv')

# test with first 2 documents
health_data_sample = health_data.head(2) 
work_data_sample = work_data.head(2)  

health_data_sample['text'] = health_data_sample['Question'].fillna('') + ' ' + health_data_sample['Answer'].fillna('')
work_data_sample['text'] = work_data_sample['Theme'].fillna('') + ' ' + work_data_sample['Content'].fillna('')

# chromadb is a vector database (it stores and is used to query vector embeddings = numerical representations of the data)
# client is used to interact with the database
client = chromadb.PersistentClient(path="./chroma_db")

# chroma stores the data in collections (container for vector embeddings) -- these containers will hold the embeddings for the different data sources
health_collection = client.get_or_create_collection(name="health_docs")
work_collection = client.get_or_create_collection(name="work_docs")

# create func to call on both collections
def add_data_to_collection(collection, data):
    for idx, row in data.iterrows():
        try:
            # reference on the embedding model used: https://ollama.com/library/mxbai-embed-large
            # basically embed the documents / csv data
            response = ollama.embeddings(model="mxbai-embed-large", prompt=row['text'])
            embeddings = response["embedding"]
            # get the embeddings and add it to the collection
            collection.add(
                # collection entry is uses unique id, the embedding and the original doc/ text
                ids=[str(idx)],  
                embeddings=embeddings,
                documents=[row['text']]  
            )
        except ollama._types.ResponseError as e:
            print(f"err on index {idx}: {e}")

# add
add_data_to_collection(health_collection, health_data_sample)
add_data_to_collection(work_collection, work_data_sample)

# get the relevant doc from the correct collection
def get_relevant_document(query, category):
    try:
        # now we embed the query/user prompt with the same embedding model
        response = ollama.embeddings(model="mxbai-embed-large", prompt=query)
        # decide which collection based on category of query
        collection = health_collection if category == "health" else work_collection
        # query from the collection we created for the top result 
        results = collection.query(query_embeddings=[response["embedding"]], n_results=1)
        return results['documents'][0][0] if results['documents'] else None
    except ollama._types.ResponseError as e:
        print(f"err querying: {e}")
        return None

# gen answer based on query and category (to find the correct collection)
def generate_answer(query, category):
    # heres how you can generate a response with the model - no RAG
    output_before_rag = ollama.generate(
        # im using phi, u can change this to llama or mistral
        model="phi:latest",  
        prompt=f"Respond to this question: {query}"
    )
    response_before_rag = output_before_rag['response']

    # get the relevant doc for this query
    relevant_document = get_relevant_document(query, category)
    if relevant_document is None:
        return f"sorry, no find a relevant document. Model's response before RAG: {response_before_rag}"

    # now generate using rag
    output_after_rag = ollama.generate(
        model="phi:latest", 
        prompt=f"Using this information: {relevant_document}. Respond to this question: {query}"
    )
    response_after_rag = output_after_rag['response']
    
    # we return both to compare
    return {
        "Before RAG Response": response_before_rag,
        "After RAG Response": response_after_rag
    }

# heres an example of how to use this 
# put ur query here
user_query = "What do I need to do to apply for MSP coverage in B.C.?"

# need a way to extract these categories and potentially will have a 2 step pipeline where model first flags category then we search the corresponding collection
category = "health"
# call the generate_answer with this query
responses = generate_answer(user_query, category)

# print
print("User Query:", user_query)
print("Response Before RAG:", responses["Before RAG Response"])
print("Response After RAG:", responses["After RAG Response"])


'''
Example Trial 1:

User Query: What do I need to do to apply for MSP coverage in B.C.?
Response Before RAG:  To apply for MSP (Mental Stress Program) coverage in British Columbia, you will first need to contact your local health authority. They can provide information on how to apply and the documentation required. You may also be able to find this information online through their website or by calling their customer service line. Once you have gathered all necessary documents, such as medical records and supporting letters from your doctor, you can submit an application for MSP coverage. The application process typically involves providing details about your medical condition and how it affects your ability to work. It is important to note that not everyone who applies will be approved for coverage, so be sure to provide all necessary information to support your claim.

Response After RAG:  To apply for MSP coverage in B.C., you will need to submit a completed application form along with supporting documents, such as proof of residence, name and birthdate of each dependent, and any applicable information about your employment or income. You can apply online through the BC Ministry of Health website or by calling 1-888-853-8822.  
'''