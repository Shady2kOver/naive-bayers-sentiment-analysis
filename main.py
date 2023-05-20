import pickle
import os

from flask import Flask, render_template, request

from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

MODEL_DIR =  'models'


    

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Get the text input from the form
        text = request.form['text']
        # Perform sentiment analysis and get the prediction
        prediction = perform_sentiment_analysis(text)

    return render_template('index.html', prediction=prediction)

def perform_sentiment_analysis(text):
    # Perform the sentiment analysis and return the prediction
    
    # Load the saved model
    with open('models\model.pkl', 'rb') as file:
        saved_model = pickle.load(file)

    # Load the CountVectorizer used for training
    vectorizer_path = os.path.join(MODEL_DIR, r'vectorizer.pkl')

    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)


        # Vectorize the input text
    text_vec = vectorizer.transform([text])

    # Make predictions
    prediction = saved_model.predict(text_vec)[0]

    # Return the prediction (1 for positive, 0 for negative)
    return 'Positive' if prediction == 1 else 'Negative'


if __name__ == '__main__':
    app.run()
