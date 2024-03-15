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

## 2. Stock Price Prediction

The project aims to forecast the next 3 days of stock prices by employing a Stacked LSTM model trained on a dataset comprising the preceding 100 days of historical stock data. In addition to the predictive model, a Streamlit application is developed to facilitate stock price analysis and inter-stock comparisons. A recorded demonstration of the application's functionality is also included in the repository.

#### Methodology:
1. The project begins by gathering historical stock price data using web scrapping.
2. This data is used to train a Stacked LSTM model, which is chosen for its ability to capture complex temporal patterns in sequential data.
3. The model is then fine-tuned and evaluated using appropriate performance metrics.
4. Simultaneously, a Streamlit application is developed to provide users with an intuitive interface for exploring stock prices and comparing different stocks.

#### Results:
1. The Stacked LSTM model demonstrates promising performance in predicting the next 3 days of stock prices.
2. Evaluation metrics such as root mean squared error (RMSE) indicate the model's effectiveness in capturing stock price trends.
3. Furthermore, the Streamlit application offers users a seamless experience in visualizing and analyzing stock data, allowing for easy comparison between various stocks.

#### Conclusion:
The project successfully showcases the potential of Stacked LSTM models for stock price prediction and demonstrates the utility of Streamlit applications in facilitating stock market analysis. By providing users with a user-friendly interface and accurate predictions, the project contributes to enhancing decision-making processes in financial markets. Additionally, the inclusion of a recorded demonstration ensures transparency and accessibility, further enhancing the project's value and usability.
