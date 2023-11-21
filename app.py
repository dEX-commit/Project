import streamlit as st
import pickle
import nltk 
import string
import sklearn

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
#creating list
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
        #we don't copy list like text = y but as test = y[:] called cloan
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    #stemming(converting words into base words or root words)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl' , 'rb'))

st.title("Email/SMS spam classifier")

input_sms = st.text_area("Enter the message")
if st.button("PREDICT"):

    #1.preprocess
    transform_sms = transform_text(input_sms)
    #2.Verctorize
    vector_input = tfidf.transform([transform_sms])
    #3.predict
    result = model.predict(vector_input)[0]
    #Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")
