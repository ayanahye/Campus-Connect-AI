import csv

def convert_txt_to_csv(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    qa_pairs = content.split('Q:')[1:]  
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Question', 'Answer'])  
        
        for pair in qa_pairs:
            parts = pair.split('A:')
            question = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else ''
            writer.writerow([question, answer])

convert_txt_to_csv('data.txt', 'data-no-sources.csv')

'''
Method:
I reviewed two government resources on the Medical Services Plan (MSP) for Canada and health fees for international students, focusing on the questions and answers. I then extracted relevant text and organized it into a document with one bullet point for each statement, followed by a second bullet for any additional details/context. Using the GPT-4 model, I transformed these statements into question-answer pairs for the RAG, applying the same prompt to both pre-existing Q&A and unformatted information

Prompt Used:
Please take this information collected from the Government of BC Canada website on MSP (Medical Services Plan) and Health fees for internation students, and format them as question / answer pairs for a RAG Chatbot for assisting international students new to B.C. Ensure you only rely on the information provided and no other information. Only return the questions and answers and nothing else. If there are bullet points, combine them into 1 line to make it easier to process. Put bullet points on the same line (sentence format only).
'''