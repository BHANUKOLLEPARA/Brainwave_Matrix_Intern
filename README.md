# Fake News Detection

## Introduction
Fake News Detection is a machine learning-based project designed to classify news articles as real or fake. This project leverages Logistic Regression and Decision Tree Classification to build predictive models that analyze and classify news articles based on their content.

## Features
- Preprocessing text data for machine learning.
- Two classification models:
  - Logistic Regression
  - Decision Tree Classifier

## Prerequisites
Before running the project, ensure you have the following installed:

- Python 3.7 or later

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   ```

## Dataset
The project uses a dataset containing labeled news articles. You can use publicly available datasets like the [Fake News Dataset](https://www.kaggle.com/c/fake-news) from Kaggle. Make sure the dataset contains:
- News content
- Labels indicating whether the news is real or fake

Place the dataset in the `data/` folder.

## Workflow
1. **Data Preprocessing:**
   - Clean the text data by removing punctuation, stop words, and performing stemming/lemmatization.
   - Convert text data to numerical features using techniques like TF-IDF or CountVectorizer.

2. **Model Training:**
   - Train Logistic Regression and Decision Tree models on the preprocessed dataset.

3. **Evaluation:**
   - Evaluate the performance of both models using metrics like accuracy, precision, recall, and F1 score.

4. **Prediction:**
   - Use the trained models to classify new news articles.

## Results
The performance of the models is as follows:

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 95%      | 94%       | 96%    | 95%      |
| Decision Tree        | 92%      | 91%       | 93%    | 92%      |

*Note: Results may vary based on the dataset and parameter tuning.*

## Technologies Used
- Python
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Seaborn

## Contributing
Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.

## Acknowledgments
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news)

--- 
