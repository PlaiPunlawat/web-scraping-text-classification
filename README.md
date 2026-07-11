# kmitl-news-scraping-text-classification

> KMITL Assignment 2 - Data Science Course
> Student: Pulawat Leecharoen (ปุลวัชร ลี้เจริญ) | ID: 62070256

## Overview

This project implements an end-to-end NLP workflow: scrape news articles (Task 1), preprocess text, build TF-IDF features, and train/tune multiple classifiers (Logistic Regression, KNN, Multinomial Naive Bayes, Linear SVC).

Headline result from the notebook outputs:
- Overall best: **98.90% test F1** (Task 6 Optional, Heading+Content)
- Model setup: **Snowball Stemmer + tuned Linear SVC**

The notebook conclusion also reports **Snowball Stemmer** and **Lemmatizer with POS** as the strongest normalization approaches across experiments.

## Pipeline & Technologies

| Stage | Technique | Tools |
|-------|-----------|-------|
| 1. Data Collection | Web scraping, HTML parsing, DOM traversal | `requests`, `BeautifulSoup` |
| 2. Text Preprocessing | Tokenization, stemming (Porter, Snowball, Lancaster), lemmatization with POS tagging | `nltk` |
| 3. Feature Engineering | TF-IDF vectorization | `scikit-learn` |
| 4. Classification | Multi-model training and comparison | `scikit-learn` |
| 5. Hyperparameter Tuning | Randomized search with cross-validation | `RandomizedSearchCV` |
| 6. Evaluation | Confusion matrix, ROC/AUC, precision, recall, F1 | `matplotlib`, `seaborn` |

## Results

### Task 5.1 (Content): Porter Stemmer + Tuning

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Linear SVC** | **98.53%** | **98.54%** | **98.53%** | **98.53%** |
| Logistic Regression | 98.17% | 98.18% | 98.17% | 98.17% |
| Multinomial Naive Bayes | 97.44% | 97.45% | 97.44% | 97.44% |
| K-Nearest Neighbors | 96.34% | 96.34% | 96.34% | 96.34% |

### Overall Best (Task 6 Optional, Heading+Content)

- **Model:** Snowball Stemmer + tuned Linear SVC
- **Test Accuracy:** **98.90%**
- **Test F1:** **98.90%**

## Data

The articles were scraped from the KMITL course archive site at assignment time. That source may no longer be online, so a snapshot is committed in `data/`:

- `data/categories.txt`
- `data/articles_content.txt`
- `data/articles_heading_content.txt`

Reproducibility notes:
- Task 1 (scraping) is kept as reference code.
- Tasks 2-7 are reproducible directly from the committed snapshot.

## Getting Started

### Prerequisites

- Python 3.7+
- Google Colab or local Jupyter environment

### Installation

```bash
pip install -r requirements.txt
```

### Run

Open the notebook:

```bash
jupyter notebook news_text_classification.ipynb
```

Execution guidance:
- In Google Colab, uncomment the setup cell near the top to clone the repo and enter the project directory.
- For normal reproducible runs, start from **Task 2**.
- Run **Task 1** only if you want to re-scrape (source site may be offline).

## Dependencies

See `requirements.txt` for the full list. Key libraries:

| Category | Libraries |
|----------|-----------|
| Web Scraping | requests, beautifulsoup4 |
| Data Handling | pandas, numpy |
| NLP | nltk |
| Machine Learning | scikit-learn |
| Visualization | matplotlib, seaborn |

## License

MIT License - Copyright (c) Pulawat Leecharoen
