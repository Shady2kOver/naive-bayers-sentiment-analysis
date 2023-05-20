## Description

Sentiment Analyzer predicts the sentiment (positive or negative) for online reviews . It uses the [Naive Bayes algorithm](https://www.javatpoint.com/machine-learning-naive-bayes-classifier) to analyze text input and determine whether it expresses a positive or negative sentiment. 

The model was trained on an [IMDB dataset of 50k movie reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download), each pre-labeled with sentiment so I didn't have too much of a problem training it

Unlike my last ML project this one has a web-based UI built with Flask so you can test it in real time. 

**Uses:** 

 - Python 
 - Flask (for web app)
 - HTML/CSS
 - Bootstrap
 - NLTK
 - Scikit-Learn

## Usage

1.  Clone the repository: `git clone https://github.com/Shady2kOver/naive-bayers-sentiment-analysis.git`
2.  Install the required dependencies: `pip install -r requirements.txt`
3.  Run the application: `python main.py`
4.  Redirect yourself through the output to the localhost web-page
5.  Enter the review text in the provided text area and click the "Analyze Sentiment" button
6.  The predicted sentiment (positive or negative) will be displayed below the button.

## Details
**Negation Handling**: An issue (which still isn't efficiently handled) that I faced in this was handling negations, when you tokenize words and omit stop-words you can run into a few problems .

Say there's two reviews as follows :

> I dislike this product! It has absolutely ruined my life.

> I don't dislike this product at all.

In the second review, the word *don't* negates the word dislike , the second review is showcasing a positive sentiment but the word dislike is known by the algorithm after training to be associated with negative sentiments, so the algorithm might mark this review as negative.

I've incorporated very simple negation handling where negation words like "not" "neither" "nor" negate the word after them and get tokenized as "not_good" for example, but this still doesn't handle cases like where suffixes like "*-less*" are used.

I've also used [word lemmatization](https://www.nltk.org/_modules/nltk/stem/wordnet.html) prior to negation to make the training process more smooth and not require too many word variations or inflections.


