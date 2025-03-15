import csv
import re
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# tokenize, analyze 
nlp = spacy.load("en_core_web_sm")

with open('collected-text.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# these themes were created by passing a list of frequent words (result of running get-frequent-words.py and the sources in the collected-text.txt file to ChatGPT and asking it to create 20 themes with these keywords. )
themes = {
    "International Students Resources": ["resources", "international students", "school", "education", "student"],
    "Post-Secondary Students Support": ["post-secondary", "college", "university", "student", "education", "learn", "programs"],
    "English Language Learning Resources": ["english", "language learning", "ESL", "learn English", "study", "education"],
    "Settlement Services Overview": ["settlement services", "newcomers", "immigrants", "integration", "canadian services", "help", "services"],
    "Settlement Services for Permanent Residents": ["permanent residents", "PR", "permanent residency", "Canada", "settlement", "program", "services"],
    "Settlement Services for Temporary Residents": ["temporary residents", "work permit", "temporary work", "temporary status", "job", "work", "employment"],
    "Settlement Services for Refugees": ["refugees", "asylum", "refugee claim", "BC Safe Haven", "newcomers", "help", "services"],
    "Pre-arrival Services for Permanent Residents": ["pre-arrival", "permanent residents", "moving to Canada", "immigration services", "work", "employment", "programs"],
    "Support for Skilled Immigrants in BC": ["skilled immigrants", "career paths", "workBC", "skilled professionals", "work", "employment", "job"],
    "WorkBC Employment Services": ["WorkBC", "employment", "job search", "career services", "workers", "services", "program"],
    "Work Permits for International Students": ["study permit", "work permit", "student work", "campus jobs", "off-campus work", "permit", "work", "students"],
    "Finding Work in British Columbia": ["finding work", "job market", "work in BC", "employment opportunities", "job", "work", "canada"],
    "Certification for Work in BC": ["certification", "work certification", "BC certifications", "regulated professions", "workers", "job", "program"],
    "Working in British Columbia": ["working in BC", "work experience", "employment laws", "labor rights", "working", "workers", "conditions"],
    "International Student Work Rights (On-Campus)": ["on-campus work", "student work rights", "on-campus jobs", "campus", "work", "student", "employment"],
    "International Student Work Rights (Off-Campus)": ["off-campus work", "student work rights", "work outside campus", "work permit", "students", "work"],
    "Internships for International Students": ["internships", "student internships", "internship opportunities", "work", "student", "job"],
    "Extending a Study Permit": ["study permit extension", "renew study permit", "extend permit", "permit", "study", "canada"],
    "Study Permit Expiration and Renewal": ["study permit expiration", "permit renewal", "expired permit", "study permit", "permit", "canada"],
    "Study Permit Application Process": ["study permit application", "how to apply", "apply for study permit", "study permit", "apply", "permit"]
}

def extract_sections(data, themes):
    sections = {}
    parts = re.split(r"(\*\*\* Source: .+)", data)
    
    current_source = ""
    for part in parts:
        if part.startswith("*** Source:"):
            # if start w source then store this as current source to populate other rows
            # if u use this script make sure u chnage the above line to whatever you called it to indicate source
            current_source = part.strip()
        else:
            # split by blank lines, ignore blocks with < 5 words
            blocks = re.split(r'\n\s*\n', part.strip())
            for block in blocks:
                if len(block.split()) >= 5:
                    doc = nlp(block)
                    block_lower = block.lower()
                    # match the kwywords to themes
                    # return dict where keys are themes and values are lists with source and content
                    for theme, keywords in themes.items():
                        matching_keywords = sum(1 for keyword in keywords if keyword.lower() in block_lower)
                        if matching_keywords >= 2:
                            if theme not in sections:
                                sections[theme] = []
                            sections[theme].append((current_source, block.strip()))
    return sections

def clean_content(content):
    content = re.sub(r"\b(last reviewed|updated)\s*:\s*\w+\s*\d{1,2},\s*\d{4}", "", content)
    content = re.sub(r"(back to top|to top)", "", content)
    return content.strip()

sections = extract_sections(data, themes)
csv_data = []

for theme, content_list in sections.items():
    for source, content in content_list:
        cleaned_content = clean_content(content)
        if cleaned_content:  
            csv_data.append([theme, source, cleaned_content])

dataa = pd.DataFrame(csv_data, columns=["Theme", "Source", "Content"])
dataa = dataa.drop_duplicates(subset=["Content"], keep="first")
# note that there will be duplicate content for different themes
# i am leaving this for now but may drop duplicates if its not working out.

with open('work-and-education-data.csv', mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Theme", "Source", "Content"])
    writer.writerows(csv_data)

