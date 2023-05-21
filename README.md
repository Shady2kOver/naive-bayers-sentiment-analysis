# Description

Sentiment Analyzer predicts the sentiment (positive or negative) for online reviews . It uses the [Naive Bayes algorithm](https://www.javatpoint.com/machine-learning-naive-bayes-classifier) to analyze text input and determine whether it expresses a positive or negative sentiment. 

The model was trained on an [IMDB dataset of 50k movie reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download), each pre-labeled with sentiment so I didn't have too much of a problem training it

Unlike my last ML project this one has a web-based UI built with Flask so you can test it in real time. 

## **Uses:**

 - Python 
 - Flask (for web app)
 - HTML/CSS
 - Bootstrap
 - NLTK
 - Scikit-Learn

# Usage

1.  Clone the repository: `git clone https://github.com/Shady2kOver/naive-bayers-sentiment-analysis.git`
2.  Install the required dependencies: `pip install -r requirements.txt`
3.  Run the application: `python main.py`
4.  Redirect yourself through the output to the localhost web-page
5.  Enter the review text in the provided text area and click the "Analyze Sentiment" button
6.  The predicted sentiment (positive or negative) will be displayed below the button.

# Details

## ** Naive Bayes Classifier**:

**Bayes' theorem** states that the probability of an event A given the occurrence of event B is equal to the probability of event B given event A multiplied by the probability of event A, divided by the probability of event B. 

<sub>*Note that this is a simple explanation for 2 events, when considering many events we will take in the total probability in the denominator*</sub>

This can be represented as follows - ![Bayes Theorem](https://miro.medium.com/v2/resize:fit:1400/1*LB-G6WBuswEfpg20FMighA.png)


Now this classifier is called **"Naive Bayes"** classifier as it uses bayes theorem with a slight assumption, The classifier assumes that the presence or absence of a particular feature in a class is independent of the presence or absence of any other feature. 
In other words, it assumes that the features are conditionally independent given the class.

This assumption is considered "naive" because it oversimplifies the relationships between features. In reality, many features are often correlated or dependent on each other to some extent. 

However, despite this simplification, Naive Bayes has been found to work well in many practical applications and can achieve good results, especially when the independence assumption is reasonably satisfied or when the dependencies between features are not critical for accurate classification. Furthermore, it can be easily implemented through the `scikit-learn` library








## **Negation Handling**:

An issue (which still isn't efficiently handled) that I faced in this was handling negations, when you tokenize words and omit stop-words you can run into a few problems .

Say there's two reviews as follows :

> I dislike this product! It has absolutely ruined my life.

> I don't dislike this product at all.

In the second review, the word *don't* negates the word dislike , the second review is showcasing a positive sentiment but the word dislike is known by the algorithm after training to be associated with negative sentiments, so the algorithm might mark this review as negative.

I've incorporated very simple negation handling where negation words like "not" "neither" "nor" negate the word after them and get tokenized as "not_good" for example, but this still doesn't handle cases like where suffixes like "*-less*" are used.

I've also used [word lemmatization](https://www.nltk.org/_modules/nltk/stem/wordnet.html) prior to negation to make the training process more smooth and not require too many word variations or inflections.


