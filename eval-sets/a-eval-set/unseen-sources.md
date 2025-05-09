# Evaluation of Unseen Data in Our RAG System

Disclaimer - although i called seen and unseen data, these questions may have appeared in the training set but we cannot verify. By seen and unseen i am referring to whether its in our documents.

Note: This dataset is primarily designed to evaluate the model's ability to provide precise, up-to-date, and reliable anwers for international student queries (e.g. question answering) using its pretrained knowledge and our docs.

Some of the questions are intentionally a bit ambiguous or multi-faceted to test how well the model could handle these cases.

Our process involves evaluating our RAG system on both:

- **Seen data**: Examples included in the documents  
- **Unseen data**: Examples not present in the documents 

---

## Purpose of This File

This document outlines the **sources and justification** for the unseen data used in evaluation.

The questions were **manually selected** by reviewing FAQs from various official online sources (listed below). The answers were adapted from these sources, with key points extracted that we believe the model should be able to recognize based on the provided data. Please note, this set is currently incomplete and may be updated in the future.

---

## Source Attribution

The following questions and modified answers were derived from the listed sources in this order:

**[1], [1], [1], [4], [4], [4], [4], [5], [5]**  
*(Last one created by us)*

---

## References

**[1]** https://issbc.org/our-resources/general-questions-and-answers/  
**[2]** https://newtobc.ca/settlement-information-for-newcomers/settling-in-bc-questions-and-answers/  
**[3]** https://students.ubc.ca/international-student-guide/international-immigration-health-insurance-faq/#travel-canada-start-term  
**[4]** https://www.sfu.ca/students/isap/faqs-international/travel-to-canada/what-documents-should-i-carry-when-i-travel-to-canada--.html  
**[5]** https://www.educanada.ca/why-canada-pourquoi/faq.aspx?lang=eng
