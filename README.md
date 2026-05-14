# Web Scraping & Text Classification

> KMITL Assignment 2 — Data Science Course  
> Student: Pulawat Lee-Charoen (ปุลวัชร ลี้เจริญ) | ID: 62070256

## Overview

This project implements a complete NLP pipeline — from scraping raw news articles to classifying them with tuned machine learning models. The best model achieves **98.53% accuracy** on the test set.

## Pipeline & Technologies

| Stage | Technique | Tools |
|-------|-----------|-------|
| 1. Data Collection | Web scraping, HTML parsing, DOM traversal | `requests`, `BeautifulSoup` |
| 2. Text Preprocessing | Tokenization, stemming (Porter, Snowball, Lancaster), lemmatization with POS tagging | `nltk` |
| 3. Feature Engineering | TF-IDF vectorization | `scikit-learn` |
| 4. Classification | Multi-model training & comparison | `scikit-learn` |
| 5. Hyperparameter Tuning | Randomized search with cross-validation | `RandomizedSearchCV` |
| 6. Evaluation | Confusion matrix, ROC/AUC, precision, recall, F1 | `matplotlib`, `seaborn` |

## Best Results (Test Set — Porter Stemmer + Tuning)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Linear SVC** | **98.53%** | **98.54%** | **98.53%** | **98.53%** |
| Logistic Regression | 98.17% | 98.18% | 98.17% | 98.17% |
| Multinomial Naive Bayes | 97.44% | 97.45% | 97.44% | 97.44% |
| K-Nearest Neighbors | 96.34% | 96.34% | 96.34% | 96.34% |

## Key Skills Demonstrated

- Data collection & web scraping pipelines
- NLP text preprocessing techniques
- Supervised classification (Logistic Regression, KNN, Naive Bayes, SVM)
- Model selection & hyperparameter optimization
- Statistical evaluation & visualization

## Getting Started

### Prerequisites

- Python 3.7+
- Google Colab (recommended) or a local Jupyter environment

### Installation

```bash
pip install -r requirements.txt
```

### Running

Open the notebook in Google Colab or Jupyter:

```bash
jupyter notebook "1_62070256_ปุลวัชร_Assign2.ipynb"
```

> **Note:** The notebook uses `gdown` to fetch training data from Google Drive. Ensure you have internet access when running.

## Dependencies

See [requirements.txt](requirements.txt) for the full list. Key libraries:

| Category | Libraries |
|----------|-----------|
| Web Scraping | requests, beautifulsoup4 |
| Data Handling | pandas, numpy |
| NLP | nltk |
| Machine Learning | scikit-learn |
| Visualization | matplotlib, seaborn |

## License

This project is part of coursework at King Mongkut's Institute of Technology Ladkrabang (KMITL).
