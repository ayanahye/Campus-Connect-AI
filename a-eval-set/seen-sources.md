# Evaluation of Seen Data in Our RAG System

Disclaimer - although i called seen and unseen data, these questions may have appeared in the training set but we cannot verify. By seen and unseen i am referring to whether its in our documents.

Note: This dataset is primarily designed to evaluate the model's ability to provide precise, up-to-date, and reliable anwers for international student queries (e.g. question answering) using its pretrained knowledge and our docs.

Testing on seen data is mostly for accuracy and testing on unseen is to see how well it can generalize.

Our process involves evaluating our RAG system on both:

- **Seen data**: Examples included in the documents 
- **Unseen data**: Examples not present in the documents 

---

## Purpose of This File

This document outlines the **sources and justification** for the seen data used in evaluation.

The questions were carefully selected by reviewing the data from [2] which provides a vareity of FAQ from international students, overlapping with the underlying themes of our data. We then adapted the answers, incorporating additional context from our collected data. Key points were extracted to evaluate the model's ability to accurately identify and present these points. Please note, this set is currently incomplete and may be updated in the future.

## References

**[2]** https://newtobc.ca/settlement-information-for-newcomers/settling-in-bc-questions-and-answers/  