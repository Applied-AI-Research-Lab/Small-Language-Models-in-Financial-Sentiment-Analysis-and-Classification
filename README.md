# Explainable AI for Business Intelligence: Small Language Models in Financial Sentiment Analysis and Classification

## Article
* **Journal**: pending
* **Title**: Explainable AI for Business Intelligence: Small Language Models in Financial Sentiment Analysis and Classification
* **DOI**: pending

## Authors
* **Adj. Asst. Prof. Konstantinos I. Roumeliotis | University of Peloponnese**
* **Asoc. Prof. Dionisis Margaris | University of Peloponnese**
* **Asoc. Prof. Dimitris Spiliotopoulos | University of Peloponnese**
* **Prof. Nikolaos D. Tselikas | University of Peloponnese**

## Abstract
Financial sentiment analysis is a cornerstone of business intelligence (BI), converting
market text into structured decision signals. Small language models (SLMs) promise
classification with fluent, natural-language explanations, but production deployment
faces a tension: pipelines must remain stable and auditable while explanation interfaces
evolve with every new release, and re-fine-tuning is impractical. We propose a decoupled
multi-model framework resolving this tension: three transparent TF-IDF classifiers
(Logistic Regression, Linear SVM, XGBoost) form an advisory panel whose predictions,
confidence, entropy, and lexical evidence, with the verbatim sentence, are passed
zero-shot to Qwen3.5-4B and Gemma3-4B (both also QLoRA fine-tuned for comparison), an
explainable AI (XAI) layer that emits a sentiment label, a natural-language explanation,
and a business recommendation. Any newer SLM can replace it without classifier
retraining, requiring only re-validation of a frozen evaluation set. On the Financial
PhraseBank benchmark, zero-shot Qwen3.5-4B attains 94.41% accuracy (macro F1 0.9348), the
highest of seven systems, significantly above the best classical model (exact McNemar
p = 0.021), yet statistically indistinguishable from the top fine-tuned systems, which
hold the highest macro F1 (Gemma3-4B, 0.9377). Input ablations confirm both information
sources are necessary, and zero-shot accuracy is decoding-seed robust (98.8% label
stability). A validation-selected fine-tuning grid establishes a 97.35%/96.76% ceiling,
indistinguishable from domain-specific FinBERT, positioning zero-shot operation at
near-ceiling accuracy and zero training cost where task training is undesirable. A
blinded two-expert evaluation bounds explanation quality (faithfulness 4.72/5), yet only
49.2% of outputs would enter an internal report unchecked.

## Keywords
Financial Sentiment Analysis, Explainable AI, Business Intelligence, Small Language
Models, Multi-Model Framework, Zero-Shot Inference, TF-IDF