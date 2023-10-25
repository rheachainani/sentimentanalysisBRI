import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

df = pd.read_csv("Joint_BRI_Sentiment.csv")
balanced_df = pd.read_csv('Balanced_BRI_data.csv')
model = pickle.load(open('model.pkl', 'rb'))

def main():

    st.set_page_config(layout="wide", page_title="BRI Sentiment Analysis", page_icon = Image.open('icon1.jpeg'))

    action = st.sidebar.radio('Select a section to check out',('About BRI','Predict Sentiment of a Text','View the Trends over the years'))

    if (action == 'About BRI'):
        st.title("What is the Belt & Road Initiative ?")
        st.write("China’s Belt and Road Initiative (BRI), sometimes referred to as the New Silk Road, is one of the most ambitious infrastructure projects ever conceived. Launched in 2013 by President Xi Jinping, the vast collection of development and investment initiatives was originally devised to link East Asia and Europe through physical infrastructure. In the decade since, the project has expanded to Africa, Oceania, and Latin America, significantly broadening China’s economic and political influence.")
        st.write("Some analysts see the project as an unsettling extension of China’s rising power, and as the costs of many of the projects have skyrocketed, opposition has grown in some countries. Meanwhile, the United States shares the concern of some in Asia that the BRI could be a Trojan horse for China-led regional development and military expansion. President Joe Biden has maintained his predecessors’ skeptical stance towards Beijing’s actions, but Washington has struggled to offer participating governments a more appealing economic vision.")
        st.write("China’s overall ambition for the BRI is staggering. To date, 147 countries—accounting for two-thirds of the world’s population and 40 percent of global GDP—have signed on to projects or indicated an interest in doing so.")
        
        image = Image.open('bri_map.png')
        st.image(image, use_column_width = True)     


    if (action == 'Predict Sentiment of a Text'):

        st.title('Predict Sentiment of a Text')

        text = st.text_input('Enter text')

        if st.button("Predict", help="Click here to predict the sentiment of text"):

            tfidf = TfidfVectorizer()
            text_feature = tfidf.fit_transform(balanced_df.tokenized_text)
            
            text_feature = tfidf.transform([text])

            prediction = model.predict(text_feature)
            #st.write(prediction)

            if prediction=='Neutral':
                st.success(f'The sentiment of the text is : {prediction[0]}', icon="😐")
            elif prediction=='Positive':
                st.success(f'The sentiment of the text is : {prediction[0]}', icon="🙂")
            else:
                st.success(f'The sentiment of the text is : {prediction[0]}', icon="☹️")


    if (action == 'View the Trends over the years'):

        st.title('The Trend in Sentiments regarding BRI')
        start_year = st.slider('From (year)', 2014, 2022, 2014)
        end_year = st.slider('To (year)', 2015, 2023, 2023)

        if (start_year < end_year):

            filtered_data = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
            sentiment_by_year = filtered_data.groupby('year')['BRIscore'].mean()

            years = sentiment_by_year.index.to_numpy()
            sentiments = sentiment_by_year.values

            fig, ax = plt.subplots()
            ax.plot(years, sentiments, marker='o', linestyle='-')
            ax.set_xlabel("Year",color='white')
            ax.set_ylabel("Average Sentiment",color='white')
            plt.tick_params(axis='x', labelcolor='white')
            plt.tick_params(axis='y', labelcolor='white')
            ax.set_title(f"Sentiments regarding BRI from {start_year} to {end_year}",color='white')
            ax.set_facecolor('black')
            fig.patch.set_alpha(0.0)
            st.pyplot(fig)

        else : 
            st.write('Start year must be before End year. Please try again.')


if __name__ == '__main__':
    main()
