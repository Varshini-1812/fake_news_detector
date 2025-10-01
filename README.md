# fake_news_detector
DistilBERT-based fake news classification mode
# 📰 Fake News Detector

This project uses a BERT-based model to detect fake news headlines and articles. It leverages two datasets—real and fake news—hosted on Google Drive and integrates seamlessly with Google Colab for easy experimentation.

## 🚀 Quick Launch

Run the notebook instantly on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Varshini-1812/fake_news_detector/blob/main/Fake_news_detector.ipynb)


## 📁 Dataset Access

Due to GitHub's file size limits, the datasets are hosted on Google Drive and automatically downloaded when the notebook runs.

- [Fake.csv](https://drive.google.com/file/d/11pL5fIIFMh4D2YIDJKmYkRA9w7vIx8u_/view?usp=sharing)
- [True.csv](https://drive.google.com/file/d/1WEDWvilWL-smeovIT96B6xJ0TXvsMrA8/view?usp=sharing)

## 🧠 Model Overview

- Uses HuggingFace's BERT for text classification
- Preprocesses news articles using tokenization and attention masks
- Trains on labeled fake and real news data
- Outputs prediction probabilities and classification labels

## 📦 Requirements

- Python 3.8+
- `transformers`, `torch`, `pandas`, `sklearn`, `gdown`

Install dependencies with:

```bash
pip install -r requirements.txt
