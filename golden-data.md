# Evaluation of RAG Systems + Presentation Outline

How do we evaluate our RAG system?  

We want to measure how well it finds the right information, writes good answers, and performs as a complete system.

---

## Why Do We Need a Benchmark?

A benchmark is a test for the system. (A standardized test that other people could also use to compare systems in a consistent way).

We need it to:

- Evaluate retrieval performance
- Evaluate answer generation quality
- Assess system as a whole

---

## How Do We Create a Benchmark?

To build a benchmark, we need:

1. A set of realistic questions  
2. Reference answers (expected/correct answers)  
3. A document source or retrieval corpus  
4. Evaluation metrics

Metrics depend on what we want to evaluate:
- **Retrieval**: how well relevant documents are found
- **Generation**: how good the written responses are

---

## What Should a Benchmark Evaluate?

For example, CRAG-RAG evaluates:
- Text continuation
- Question answering
- Hallucination correction
- Multi-document summarization [3]

---

## Types of Evaluation

There are three common types:

1. **Reference-Free**  
   No gold answers. Evaluation focuses on how appropriate the answer seems based on the input question alone. [2].

2. **Synthetic Dataset**  
   Auto-generated Q&A pairs, typically created using LLMs.  
   Advantage: Scalable. Limitation: Often lacks real-world authenticity [2].

3. **Golden Dataset**  
   Manually curated, high-quality Q&A pairs.  
   Advantage: Highest reliability and task relevance [2].

---

## Why Use a Golden Dataset?

- Ensures consistent, high-quality evaluation  
- Matches real user needs and queries

A well-designed golden dataset contains: [1]
- Realistic user questions
- Ground truth answers
- Optional: supporting context

Note: Around 100 Q&A pairs is considered reasonable [1].

---

## How Do We Know If Our Benchmark Is Good? 

- Human review: Are scores aligned with expert judgment? [1] 
- LLM reviewer: Can a model provide reliable automated evaluation? [1]

---

## What Should We Measure?

---

### Answer Quality

Core dimensions [3]:
- **Relevance** – Is the answer related to the question?
- **Accuracy** – Are the facts correct / are the relevant docs identified actually scored higher than irrelevant ones?
- **Faithfulness** – Does the answer reflect retrieved documents?

Additional aspects [1]:
- **Groundedness** – Is the answer supported by evidence?  (making sure that these claims are substantiated by the context in documents.)
- **Coherence** – Does the response read logically and fluently?

---

### Retrieval Evaluation

Measures how well the system retrieves useful documents:

- **Relevance**: Are the retrieved documents actually helpful for answering the query?
- **Ranking Quality**: Are the most relevant documents retrieved early?

#### Examples of Retrieval Evaluation Metrics

- **Non-Rank-Based Metrics** (binary outcomes: relevant = 1, not = 0): [3]
  - **Accuracy**: (Correct predictions / Total predictions)
  - **Precision**: (Relevant instances / Retrieved instances)
  - **Recall**: (Relevant instances retrieved / total amount of actual relevant instances) 

- **Rank-Based Metrics** (evaluates the position of relevant results): [3]
  - **Mean Reciprocal Rank (MRR)**: Average of the reciprocal ranks of the first relevant result (how early does the first relevant document appear in the retrieved list)
  - **Mean Average Precision (MAP)**: Mean of the average precision scores for all queries (how well does the system rank multiple relevant docs for each query)

---

### Generation Evaluation

This measures the quality of the generated output.

#### 1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

- Compares n-gram overlaps between the generated and reference answers. (Measure how similar the generated text is to a reference answer by looking at how many words or word patterns are shared between them) [3]
- Common types:
  - **ROUGE-N**:
  N-gram overlap 
  - **ROUGE-L**: 
  Longest common subsequence
  - **ROUGE-W**: 
  Weighted version of ROUGE-L givens more importance to longer shared sequences

Useful for summarization and paraphrasing.

#### 2. BLEU (Bilingual Evaluation Understudy)

- Also used to see how close a machine generated sentence is to a reference sentence.
- Precision-based n-gram overlap metric, with a brevity penalty. (calculates precision of n-grams in generated translation and compares to reference translation, while penalizing answers that are too short) [3]


#### 3. BERTScore

- Uses contextual embeddings to compute similarity between generated and reference text. (basically breaks down generated response and reference response into tokens and checks how closely these match in meaning) [3]
- Based on meaning rather than just exact words

#### 5. LLM-as-Judge

LLM prompted to assess quality based on: [3]

- Relevance
- Coherence
- Fluency
- Coverage
- Level of Detail
- Diversity

---

## Other Factors to Evaluate in a RAG System [3]

- **Latency**: How fast is the response time? Important for us! 

- **Diversity**: Are the retrieved documents redundant or do they add value?  
  - Use cosine similarity between documents: lower similarity implies more diverse content.
    - Essentially we would want high similarity to the query, but low similarity between documents (however this may not be as important for us).

- **Noise Robustness**: Can the system resist misleading content?  
  - Metrics include Misleading Rate and Mistake Reappearance Rate [4].
  - Basically, how does it handle irrelevant or misleading info?

- **Negative Rejection**: Can the model say "I don’t know" when there’s insufficient information? [3]

- **Counterfactual Robustness**: Can the system detect and avoid using false evidence?  
  - Example metric: Error Detection Rate (ratio of counterfactual statements detected in retrieved info) [3]

---

## So, What Should We Use for Our Project?

Honestly, it's up to you, feel free to try anything and we can discuss results in the meeting. But please keep track of what you test so we can add it to the paper...

1. **Golden Dataset**
   - We will use 100 realistic student queries and accurate answers (let's start with 20 each for now then discuss in meeting)

3. **Some Metrics I Think Are Interesting For This**
   - Retrieval: Precision@k, Recall@k, MRR, MAP
   - Generation: BERTScore, LLM Judge scores

5. **Additional Monitoring**
   - Please track latency
   - Noise Robustness could be interesting

---

## References

[1] [The Path to a Golden Dataset](https://medium.com/data-science-at-microsoft/the-path-to-a-golden-dataset-or-how-to-evaluate-your-rag-045e23d1f13f)  
[2] [How Important is a Golden Dataset for LLM Evaluation?](https://blog.relari.ai/how-important-is-a-golden-dataset-for-llm-pipeline-evaluation-4ef6deb14dc5)  
[3] [Evaluation of Retrieval-Augmented Generation: A Survey](https://arxiv.org/pdf/2405.07437)  
[4] [RECALL: A Benchmark for LLMs Robustness against External Counterfactual Knowledge](https://arxiv.org/pdf/2311.08147)

## Side Notes:

I think our presentation + paper can focus on evaluation of RAG system specifically as well as getting relevant data and constructing our benchmark datasets.

Side side note: Of course we would want actual international students to evaluate our system (and ofc throughout the design process) but due to time constraints this might not be possible. To discuss still.

Here's my proposed skeleton:

- **Introduction** (what is RAG, why evaluation matters, how do we build and evaluate RAG for international students)

- **Building the RAG System** (data sources, retriever and generator models, example queries, prompt tuning process (manual review or similarity metric to benchmark dataset))

- **Creating the Benchmark Dataset** (why a benchmark, our process, evaluation metrics used)

- **Other Evaluation Aspects Assessed** (latency, diversity, noise robustness) (lower priority - only if time)

- **Challenges** (dataset quality and relevance, bias, evaluation challenges)

- **Conclusion and Future Work**
