import re
from collections import Counter
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_and_tokenize(text):
    words = text.lower().split()
    words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
    return words

with open('collected-text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

words = clean_and_tokenize(text)

word_counts = Counter(words)

most_common = word_counts.most_common(200)  

for word, count in most_common:
    print(f"{word}: {count}")


