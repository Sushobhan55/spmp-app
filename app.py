import flask
import pickle
from text_pipeline import text_pipeline
from scraper import get_news
import pandas as pd


app = flask.Flask(__name__, template_folder='templates')

path_to_vectorizer = 'models/vectorizer.pkl'
path_to_classifier = 'models/movement-classifier.pkl'

with open(path_to_vectorizer, 'rb') as f:
    vectorizer = pickle.load(f)

with open(path_to_classifier, 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('index.html'))

    if flask.request.method == 'POST':
        ticker = flask.request.form['ticker']
        try:
            df = get_news(ticker)

            predictions = []
            clean_text = []
            for i in range(len(df.title)):
                text = text_pipeline(df.title[i])
                clean_text.append(text)
            X = vectorizer.transform(clean_text)
            prediction = model.predict(X)
            predictions.append(prediction)

            df["prediction"] = predictions[0]

            ups = df["prediction"].value_counts()["Up"]
            downs = df["prediction"].value_counts()["Down"]
            total = ups+downs

            return flask.render_template('index.html',
                                    input_text=ticker.upper(),
                                    table=df.values.tolist(),
                                    headings=['Date', 'Time', 'Title', 'Prediction'],
                                    total=total, ups=ups, downs=downs)

        except Exception as e:
            e = "Message: No news on "+ ticker.upper() +" has been published yet."
            return flask.render_template('index.html',
                                         error=e)

@app.route("/contacts/")
def contacts():
    return flask.render_template("contacts.html")

@app.route("/about/")
def about():
    return flask.render_template("about.html")


if __name__ == '__main__':
    app.run(debug=True)
