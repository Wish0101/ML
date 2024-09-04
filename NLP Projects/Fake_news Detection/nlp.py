import re
import time
import pandas as pd
import streamlit as st
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Start = time.time()

# load the dataset

News_DF = pd.read_csv('train.csv')
# print(News_DF.head())

News_DF.fillna(' ',inplace=True)
News_DF['Content'] = News_DF['author'] + ' ' + News_DF['title']
x = News_DF.drop('label',axis=1)
y = News_DF['label']

# print(News_DF['Content'])

ps = PorterStemmer()
def Preprocess_data(content):
    cleaned_Data = re.sub('[^a-zA-Z]', ' ', content)
    lower_data = cleaned_Data.lower()
    splitted_data = lower_data.split()
    stemmed_data = [ps.stem(word) for word in splitted_data]
    stemmed_data = ' '.join(stemmed_data)
    return stemmed_data

News_DF['Content'] = News_DF['Content'].apply(Preprocess_data)

x = News_DF['Content'].values
y = News_DF['label'].values
vector = TfidfVectorizer(stop_words='english')
vector.fit(x)
x = vector.transform(x)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=
                                                       0.2,stratify=y)

model = LogisticRegression()
model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_accuracy = accuracy_score(y_train,y_train_pred)
test_accuracy = accuracy_score(y_test,y_test_pred)
print(train_accuracy)
print(test_accuracy)
print('\n')
end = time.time() - Start
print(f'{end} seconds')

st.title('Fake news detector')
input_text = st.text_input('Enter your news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    pred = model.predict(input_data)
    return pred

if input_text:
    predicted_output = prediction(input_text)
    
    if predicted_output == 1:
        st.write('Fake News')
    else:
        st.write('Real news')