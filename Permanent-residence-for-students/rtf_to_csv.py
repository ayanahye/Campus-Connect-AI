import csv 
from striprtf.striprtf import rtf_to_text

def convert_rtf_to_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        rtf_content = f.read()
        text = rtf_to_text(rtf_content)
        
    qa_pairs = text.split("Question:")[1:]  
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Question', 'Answer'])  
        
        for pair in qa_pairs:
            parts = pair.split("Answer:")
            question = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else ''
            writer.writerow([question, answer])

    print(f"Conversion complete: {output_file}")



convert_rtf_to_csv('data.rtf', 'permanent_residence_student_general.csv')