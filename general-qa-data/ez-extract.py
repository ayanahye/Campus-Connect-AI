from bs4 import BeautifulSoup

import csv

with open('raw.txt', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')

accordion_items = soup.find_all('div', class_='accordion-item')
qa_pairs = []

for item in accordion_items:
    question_tag = item.find('span', class_='accordion-button-label')
    question = question_tag.get_text(strip=True) if question_tag else 'N/A'

    answer_container = item.find('div', class_='accordion-body')
    if answer_container:
        answer = ' '.join(tag.get_text(strip=True) for tag in answer_container.find_all(['p', 'li', 'em', 'strong', 'a']))
    else:
        answer = 'N/A'

    if question != 'N/A' and answer != 'N/A' and question.strip() != '' and answer.strip() != '':
        qa_pairs.append({'question': question, 'answer': answer})

with open('faq_qa_pairs.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['question', 'answer'])
    writer.writeheader()
    writer.writerows(qa_pairs)
