from nltk.translate import meteor_score
from rouge_score import rouge_scorer
import pandas as pd
import nltk

# ignore: https://aclanthology.org/2025.naacl-long.182.pdf
# bleu and variants might not be the best here
# im using meteor for reference:  https://www.nltk.org/api/nltk.translate.meteor_score.html
# summary:
    # Aligns/matches words in the hypothesis to reference by sequentially applying exact match, stemmed match and wordnet based synonym match. In case there are multiple matches the match which has the least number of crossing is chosen.
nltk.download('punkt')

rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate_meteor_rouge(references, predictions):
    results = {"METEOR": [], "ROUGE-L": []}
    
    tokenized_references = [nltk.word_tokenize(ref) for ref in references]
    tokenized_predictions = [nltk.word_tokenize(pred) for pred in predictions]
    
    meteor_scores = [meteor_score.single_meteor_score(ref, pred) for ref, pred in zip(tokenized_references, tokenized_predictions)]
    results["METEOR"] = meteor_scores

    for ref, pred in zip(references, predictions):
        rouge_result = rouge.score(ref, pred)
        results["ROUGE-L"].append(rouge_result["rougeL"].fmeasure)

    return results

file1 = "poorvi-unseen-data.csv"
file2 = "serena-seen-data.csv"

df1 = pd.read_csv(file1)
refs1 = df1["Actual_Answer"].fillna("").tolist()
gens1 = df1["Gen_Answer"].fillna("").tolist()
scores1 = evaluate_meteor_rouge(refs1, gens1)
df1["METEOR"] = scores1["METEOR"]
df1["ROUGE-L"] = scores1["ROUGE-L"]
df1.to_csv("r_and_b/poorvi-unseen-data-scored.csv", index=False)

df2 = pd.read_csv(file2)
refs2 = df2["Benchmark Answer"].fillna("").tolist()
gens2 = df2["RAG Response"].fillna("").tolist()
scores2 = evaluate_meteor_rouge(refs2, gens2)
df2["METEOR"] = scores2["METEOR"]
df2["ROUGE-L"] = scores2["ROUGE-L"]
df2.to_csv("r_and_b/serena-seen-data-scored.csv", index=False)
