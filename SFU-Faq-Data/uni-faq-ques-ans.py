import csv

def convert_to_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    data = []
    current_link = None
    
    # Process the file line by line
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        
        # If the line doesn't contain a question mark, it's a link
        if '?' not in line:
            current_link = line
        else:
            # Split the line into question and answer
            parts = line.split('?', 1)  # Split at the first question mark
            if len(parts) == 2:
                question = parts[0].strip() + '?'  # Add the question mark back
                answer = parts[1].strip()
                data.append([question, answer, current_link])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Question', 'Answer', 'Link']) 
        writer.writerows(data)
    
    return len(data) 

input_file = 'sourcedata-folder/qa_international.txt'
output_file = 'sfu-immigration-faq.csv'  

count = convert_to_csv(input_file, output_file)
print(f"Converted {count} question-answer pairs to {output_file}")

'''
Method:
I reviewed the information provided on the official SFU websites, focusing on the questions and answers. I then extracted relevant text and organized it into a document. Using the GPT-4 model, I transformed these statements into question-answer pairs for the RAG (generated-qa.txt), applying the prompt below.

Model: ChatGPT-4o
Prompt Used:
The information provided below is taken from the official SFU websites. From this information, generate question answer pairs, with each answer being no longer than a single sentence. Please only generate the relevant question answer pairs, and no other data. Make sure there are no bullet points or other formatting applied. As a context, these question answer pairs are going to be used for a RAG application for international students as an assistance.
'''