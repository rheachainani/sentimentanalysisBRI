# Sentiment Analysis on the Belt and Road Initiative by China

Belt and Road Initiative (BRI) is an infrastructural project launched by China to connect East Asia, Europe and Africa through physical infrastructure and aims at enhancing regional connectivity.
In this project, we applied sentiment analysis to a diverse range of texts, including news articles and social media comments, to gain a comprehensive understanding of public sentiment towards the BRI. The project comprised of the various steps of the data science life cycle, from text data preprocessing, to data modeling, as well as the final model deployment using streamlit.

### Data Collection
We scraped YouTube comments on BRI related videos and combined that with the already existing dataset from Zenodo which contained headlines of Chinese newspaper starting from 2014 up till 2023. The dataset has 18191 entries of text, its recording year, source and corresponding sentiment labels and scores.

### Data Processesing
Various preprocessing steps were applied on the dataset:
1. Conversion to lower case 
2. Removing punctuations 
3. Handling emojis (for YouTube comments)
4. Removing non ascii characters (for Chinese characters in news headlines)
5. Stop word removal 
6. Aspect based sentiment assigning 
7. Lemmatization
8. Tokenization

### Data Modeling
We used Logistic Regression and got an accuracy of 71.9%.
While using the data for modeling, we removed some of the positive sentiment texts in order to balance the data and get an unbiased and robust model.

### Model Deployment
Our model was globally deployed using Streamlit and Github. https://sentimentanalysisbri.streamlit.app/
