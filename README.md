### A. The project problem statement -

This project implements an emotion classification model using logistic regression and a multi-label classification approach. 
The model is designed to predict emotions such as **anger**, **fear**, **joy**, **sadness**, and **surprise** based on textual data. 
It is built using Python, Scikit-learn, and NLTK.

[  Multi-label Emotion Detection: Given a target text snippet, predict the perceived emotion(s) of the speaker. 
Specifically, select whether each of the following emotions apply: joy, sadness, fear, anger, surprise, or disgust.
In other words, label the text snippet with: joy (1) or no joy (0), sadness (1) or no sadness (0), anger (1) or no anger (0), surprise (1) or no surprise (0).
]

### B. Following are the solution -
[ 1 ] Carried out the Dataset analysis ( respective pdf file and plots can be viewed at '/Dataset Analysis' )
- 1.1 **track-a.csv**: A sample CSV file containing labeled text data for training the emotion classification model.

[ 2 ] Code implementation ( respective code file and plots can be viewed at '/Implementation' )
- 2.1 **main.py**: The main script containing the EmotionClassifier class and all relevant functions for text preprocessing, model training, evaluation, and prediction.
- 2.2 Added Class UML & Over-all Flow diagrams

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- nltk

## Usage

### 1. Running the Model

To run the model, use the following command, providing the path to the `track-a.csv` or any other CSV file containing labeled text data:

```bash
python main.py track-a.csv
```

### 2. Text Preprocessing
(previous task_preprocess.ipynb, function def preprocess_track_a(df_dataset) have been included here )

The `EmotionClassifier` class contains a `preprocess_text` function that:

- Converts text to lowercase.
- Removes URLs, email addresses, mentions, hashtags, numbers, and punctuation.
- Tokenizes and removes stopwords.
- Applies stemming.

### 3. Model Training

The model uses the `LogisticRegression` classifier wrapped in a `MultiOutputClassifier` for multi-label classification. The model is trained on 80% of the data and tested on the remaining 20%.

### 4. Model Evaluation

The model evaluates performance based on the following metrics:

- **Hamming Loss**
- **Jaccard Score**
- **Micro and Macro average F1, Precision, and Recall**
- **Accuracy (Exact Match)**
- **Confusion Matrix**

