# Investment-Compass-Assignment

This repository contains the following projects-
1. Sentiment Analysis of News Headlines related to stocks
2. Stock Price Analysis and Prediction using Stacked LSTM model

## 1. Sentiment Analysis Project

This project aimed to analyze the sentiment of news headlines related to stocks using a Random Forest Classifier algorithm. The dataset comprised news headlines from various financial news sources. The primary objective was to classify the sentiment of each headline as either positive, negative, or neutral.

#### Methodology:

1. Data Collection: A dataset of news headlines related to stocks was collected from reputable financial news sources.
2. Data Preprocessing: The headlines were preprocessed to remove noise, such as special characters, stopwords, and punctuation, and then tokenized into words.
3. Feature Extraction: The headlines were transformed into numerical features using Count Vectorizer.
4. Model Training: A Random Forest Classifier algorithm was chosen due to its ability to handle large datasets and maintain high accuracy. The model was trained on the labeled dataset, with sentiment labels as the target variable.
5. Model Evaluation: The trained model's performance was evaluated using metrics such as accuracy, precision, recall, and F1-score.

#### Results:

1. Random Forest Classifier achieved an accuracy of 84%, indicating its effectiveness in classifying the sentiment of stock news headlines.
2. The precision, recall, and F1-score for each sentiment class were also analyzed to understand the model's performance across different sentiment categories.
3. The model demonstrated robustness in classifying sentiments despite the inherent complexity and variability of financial news headlines.

#### Conclusion:
The project successfully demonstrated the feasibility of using a Random Forest Classifier algorithm for sentiment analysis of stock news headlines. The achieved accuracy of 84% indicates that the model can effectively classify the sentiment of headlines into positive or negative categories. This analysis can be valuable for investors and traders in making informed decisions based on market sentiment extracted from news headlines. Future work may involve exploring other machine learning algorithms, fine-tuning model parameters, and incorporating additional features for further improving sentiment analysis accuracy.
