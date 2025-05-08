import os
import time
import pandas as pd
import google.generativeai as genai

api_key = ""
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

input_dir = "all_data_eval"
seen_output_file = "retrieval_seen.txt"
unseen_output_file = "retrieval_unseen.txt"

EVAL_RETRIEVAL_PROMPT_TEMPLATE = '''
You are an expert in information retrieval evaluation. Your task is to assess how well a retrieval system selected documents to answer a specific question from an international student new to British Columbia, Canada.

You will be given:
- A question posed by the student.
- A ranked list of documents retrieved by a RAG system.

Each document should be assessed for relevance to the question. A relevant document addresses the student’s question with specific, accurate, and useful information. An irrelevant document is off-topic, overly vague, or unhelpful for the student’s actual concern. Note the document does not need to answer the users question completely but it should at least be relevant to their question.

Please evaluate the retrieval on the following metrics:

**Non-Rank-Based Metrics (binary relevance):**
- Accuracy: (Correct predictions / Total predictions)
- Precision: (Relevant documents / Total retrieved documents)
- Recall: (Relevant documents retrieved / Total relevant documents assumed to exist in the set)
(Note: Assume 3 relevant documents exist in total unless otherwise stated.)

**Rank-Based Metrics:**
- Mean Reciprocal Rank (MRR): Reciprocal of the rank of the first relevant document. If no relevant document is found, MRR = 0.
- Mean Average Precision (MAP): Average of precision scores at the ranks of relevant documents, averaged across all relevant documents in the list.

Instructions:
1. For each retrieved document, indicate whether it is relevant (1) or not (0) and briefly justify your decision.
2. Then calculate the metrics listed above.
3. Your final answer should follow this CSV format:

Doc1 Relevance,Doc1 Justification,Doc2 Relevance,Doc2 Justification,...,Accuracy,Precision,Recall,MRR,MAP

Here is the user question:
{question}

Here are the retrieved documents (in order of retrieval):
{retrieved_docs}

Please begin your evaluation.
'''

batch_limit = 17
request_count = 0

for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath)

        question_col = next((col for col in df.columns if "question" in col.lower()), None)
        docs_col = next((col for col in df.columns if "retrieved" in col.lower()), None)

        if not question_col or not docs_col:
            continue  

        output_file = unseen_output_file if "unseen" in filename.lower() else seen_output_file

        with open(output_file, 'a', encoding='utf-8') as f:
            for index, row in df.iterrows():
                question = str(row[question_col]).strip()
                docs_raw = str(row[docs_col]).strip()

                if not question or not docs_raw:
                    continue

                doc_list = docs_raw.split("||")
                retrieved_docs = '\n\n'.join(f"{i+1}. {doc.strip()}" for i, doc in enumerate(doc_list) if doc.strip())

                prompt = EVAL_RETRIEVAL_PROMPT_TEMPLATE.format(question=question, retrieved_docs=retrieved_docs)

                try:
                    gemini_response = model.generate_content(prompt).text.strip()
                    request_count += 1
                except Exception as e:
                    continue

                output_block = (
                    f"\n---\n"
                    f"Question: {question}\n"
                    f"Retrieved Documents:\n{retrieved_docs}\n"
                    f"Gemini Retrieval Evaluation: {gemini_response}\n"
                )
                print(output_block)
                f.write(output_block)

                if request_count % batch_limit == 0:
                    time.sleep(60)
