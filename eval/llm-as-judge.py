import os
import time
import pandas as pd
import google.generativeai as genai

api_key = ""
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

input_dir = "all_data_eval"
seen_output_file = "gemini_seen.txt"
unseen_output_file = "gemini_unseen.txt"

EVAL_PROMPT_TEMPLATE = '''
You are an expert evaluator assessing the quality of an AI-generated response that answers questions from international students new to Canada. Use the following six criteria to evaluate the response:

Relevance: Does the response directly address the user's question and stay on topic?
Coherence: Is the response logically structured and easy to follow?
Fluency: Is the response grammatically correct and well-written?
Coverage: Does the response address all key parts of the question?
Level of Detail: Does the response go beyond generalities and provide useful, specific information?
Diversity: Does the response offer a range of perspectives, examples, or resources when appropriate?

Here is the user question:
{question}

Here is the AI-generated response:
{answer}

For each criterion, provide a score from 1 (poor) to 5 (excellent) and a one-sentence explanation of your rating. Your output should be a comma separated list like: 
Relevance Score,Relevance Explanation,Coherence Score,Coherence Explanation,Fluency Score,Fluency Explanation,Coverage Score,Coverage Explanation,Level of Detail Score,Level of Detail Explanation,Diversity Score,Diversity Explanation
'''

batch_limit = 17
request_count = 0

for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath)

        question_col = next((col for col in df.columns if "question" in col.lower()), None)
        answer_col = next((col for col in df.columns if col.lower() in ["rag response", "gen_answer", "generated answer"]), None)

        if not question_col or not answer_col:
            continue

        output_file = unseen_output_file if "unseen" in filename.lower() else seen_output_file

        with open(output_file, 'a', encoding='utf-8') as f:
            for index, row in df.iterrows():
                question = str(row[question_col]).strip()
                answer = str(row[answer_col]).strip()

                if not question or not answer:
                    continue

                prompt = EVAL_PROMPT_TEMPLATE.format(question=question, answer=answer)
                try:
                    gemini_response = model.generate_content(prompt).text.strip()
                    request_count += 1
                except Exception as e:
                    continue
                output_block = f"\n---\nQuestion: {question}\nAnswer: {answer}\nGemini Evaluation: {gemini_response}\n"
                print(output_block)
                f.write(output_block)

                if request_count % batch_limit == 0:
                    time.sleep(60)