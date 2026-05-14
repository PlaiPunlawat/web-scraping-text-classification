# Web Scraping & Text Classification

> KMITL Assignment 2 — Data Science Course  
> Student: Pulawat Lee-Charoen (ปุลวัชร ลี-เจริญ) | ID: 62070256

## Overview

This project focuses on scraping data from a website and creating models for text classification using Logistic Regression, KNeighborsClassifier, MultinomialNB, and LinearSVC.

### Pipeline

1. **Web Scraping** — Collecting news articles using `requests` and `BeautifulSoup`
2. **Text Preprocessing** — Tokenization, stemming (Porter, Snowball, Lancaster), and lemmatization with POS tagging
3. **Feature Engineering** — TF-IDF vectorization with `TfidfVectorizer`
4. **Classification** — Training and comparing four models:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Multinomial Naive Bayes
   - Linear Support Vector Classification (LinearSVC)
5. **Hyperparameter Tuning** — `RandomizedSearchCV` for optimizing model parameters
6. **Evaluation** — Confusion matrix, ROC curves, accuracy, precision, recall, F1-score

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
