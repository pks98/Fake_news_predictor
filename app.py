import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


def image():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url("https://www.torahmusings.com/wp-content/uploads/2021/05/Dice-fake-fact.jpg");
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


image()

tf = pickle.load(open('vec.pkl', 'rb'))
model = pickle.load(open('model1.pkl', 'rb'))

ps = PorterStemmer()


def text_processing(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text


st.title('Fake News Predictor')

# preprocessing

news = st.text_area('Enter the news')

if st.button('Predict'):

    transformed_news = text_processing(news)

    # vectorize
    vector_ip = tf.transform([transformed_news])

    # predict
    res = model.predict(vector_ip)[0]

    # Display

    if res == 1:
        st.error('The news is Fake')
    else:
        st.success('The news is Not Fake')
