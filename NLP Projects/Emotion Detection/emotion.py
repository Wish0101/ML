import pandas as pd
import re
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


Emotion_DF = pd.read_csv('text.csv')

# print(Emotion_DF.head())
# print(Emotion_DF.shape)

x = Emotion_DF['text']
y = Emotion_DF['label']

ps = PorterStemmer()

def data_cleaning(content):
    cleaned_data = re.sub('[^a-zA-Z\s]', '', content)
    lower_data = cleaned_data.lower()
    split_data = lower_data.split()
    stemmed_data = [ps.stem(word) for word in split_data if word]  # Remove empty strings
    stemmed_data = ' '.join(stemmed_data)
    return stemmed_data


Emotion_DF['text'] = Emotion_DF['text'].apply(data_cleaning)
x = Emotion_DF['text'].values
y = Emotion_DF['label'].values
vector = TfidfVectorizer(stop_words='english')
vector.fit(x)
x = vector.transform(x)


xTrain , xTest , yTrain , yTest = train_test_split(x,y,test_size=0.5,stratify=y)

model = LogisticRegression()
model.fit(xTrain , yTrain)

yTrainPred = model.predict(xTrain)
yTestPred = model.predict(xTest)

trainAccuracy = accuracy_score(yTrain , yTrainPred)
testAccuracy = accuracy_score(yTest,yTestPred)

print(trainAccuracy)
print(testAccuracy)

st.title('Emotion Detector')
input_text = st.text_input('Enter your emotional text')

def prediction(input_text):
    input_data = vector.transform([input_text])
    pred = model.predict(input_data)
    return pred

if input_text:
    predict_output = prediction(input_text)
    # sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).
    if predict_output == 0:
        st.write("you are sad")
    elif predict_output == 1:
        st.write("You are joyfull")
    elif predict_output == 2:
        st.write("Thats a lovely talk")
    elif predict_output == 3:
        st.write("Why are you angry")
    elif predict_output == 4:
        st.write("You are in fear")
    else:
        st.write('That was surprising')
    