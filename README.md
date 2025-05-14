# Fake News Detection System

## Overview

This project implements an end-to-end machine learning pipeline to **detect fake news** in political statements. Using the publicly available LIAR dataset, it builds, evaluates, and interprets a binary classifier that labels statements as **Real** or **Fake**, incorporating both text (TF-IDF) and linguistic features.

## Objectives

1. Preprocess and clean text data.
2. Engineer features including TF-IDF vectors, sentence length, and named entity counts.
3. Train baseline and advanced models (Logistic Regression, Random Forest, XGBoost).
4. Evaluate model performance using metrics and confusion matrix.
5. Provide interpretability with SHAP explanations.

## Repository Structure

```
fake-news-detector/
│
├── data/
│   └── liar_dataset.csv       # Raw LIAR dataset
│
├── notebooks/
│   └── fake_news_notebook.ipynb  # Jupyter notebook with code & analysis
│
├── models/
│   └── logistic_model.pkl     # Serialized best model (optional)
│
├── reports/
│   └── confusion_matrix.png   # Example plot
│
├── README.md                  # Project overview (this file)
└── requirements.txt           # Python dependencies
```

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/Cindy-00/Fake-News-Detection-System.git
   cd Fake-News-Detection-System
   ```
2. **Create & activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Download spaCy model**

   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

1. **Open** the Jupyter notebook:

   ```bash
   jupyter lab notebooks/Fake News Detection System.ipynb
   ```
2. **Run all cells** in order:

   * Data loading & cleaning
   * Feature engineering
   * Model training & evaluation
   * SHAP explainability
3. **Inspect** the outputs: classification report, confusion matrix, SHAP plots.

## Results

* **Best Model:** Logistic Regression + linguistic features
* **Accuracy:** \~60%
* **F1 (Fake):** \~0.48
* **F1 (Real):** \~0.67
* SHAP analysis identified key words influencing predictions.

## Visualizations

* Confusion matrix (`reports/confusion_matrix.png`)
* SHAP summary and force plots (in notebook outputs)

## Interpretation & Insights

* The model is more effective at identifying *real* statements than *fake* ones.
* High-impact features for *fake*: words like "illegals, hoax, rep".
* High-impact features for *real*: words like "percent, study, reported".

## Next Steps

* Experiment with transformer-based models (BERT, RoBERTa).
* Balance the dataset or apply advanced sampling techniques.
* Deploy as a Streamlit app or API for real-time classification.
