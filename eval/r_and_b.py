from nltk.translate import meteor_score
from rouge_score import rouge_scorer
import pandas as pd
import nltk
import os

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

input_dir = "all_data_eval"
output_dir = "r_and_b"
os.makedirs(output_dir, exist_ok=True)

def match_column(df_columns, candidates):
    def normalize(col):
        return col.strip().lower().replace("_", " ")
    normed_cols = {normalize(col): col for col in df_columns}
    for candidate in candidates:
        norm_candidate = normalize(candidate)
        if norm_candidate in normed_cols:
            return normed_cols[norm_candidate]
    return None

for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath)

        ref_col = match_column(df.columns, ["Actual Answer", "Benchmark Answer"])
        gen_col = match_column(df.columns, ["Generated Answer", "Gen Answer", "RAG Response"])

        if ref_col and gen_col:
            refs = df[ref_col].fillna("").tolist()
            gens = df[gen_col].fillna("").tolist()

            scores = evaluate_meteor_rouge(refs, gens)
            df["METEOR"] = scores["METEOR"]
            df["ROUGE-L"] = scores["ROUGE-L"]

            output_path = os.path.join(output_dir, filename.replace(".csv", "-scored.csv"))
            df.to_csv(output_path, index=False)
        else:
            print(f"error")
