import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("https://raw.githubusercontent.com/Sushobhan55/stock_sentiment_analysis/main/data/up_down_final")

X = df['title'].fillna(" ")

y = df['movement']

vectorizer = TfidfVectorizer()

vectorizer.fit(X)

X = vectorizer.transform(X)

model = MultinomialNB(alpha=0.025)

model.fit(X,y)

pickle.dump(vectorizer, open('models/vectorizer.pkl','wb'))
pickle.dump(model, open('models/movement-classifier.pkl','wb'))


