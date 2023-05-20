import sys
import csv
import pickle
import nltk
import numpy as np


from sklearn.model_selection import train_test_split   
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = stopwords.words('english')

TEST_SIZE = 0.4


def main():

    #Check correct usage
    if len(sys.argv)!= 2:
        print("Usage: python train_model.py [dataset.csv]")
    
    dataset_file = sys.argv[1]

    #Initialize training data to send to training function
    (reviews, sentiments) = load_dataset(dataset_file)
    
    #Preprocess the reviews to remove things that could be troubling
    new_data = preprocess_data(reviews)



    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(new_data), np.array(sentiments), test_size=TEST_SIZE
    )

    vectorizer = CountVectorizer(ngram_range=(1, 2)) #Initialize CountVectorizer

    x_train_vec = vectorizer.fit_transform(x_train) #vectorize training data features

    # Train the Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(x_train_vec, y_train)

    #Test Model
    x_test_vec = vectorizer.transform(x_test)  # Vectorize test data features
    
    predictions = classifier.predict(x_test_vec)
    
    accuracy = accuracy_score(y_test, predictions)  # Calculate accuracy

    matrix = confusion_matrix(y_test, predictions)  # Calculate confusion matrix
    print("Confusion Matrix:")
    print(matrix)

    print("Accuracy:", accuracy)

    #Save Model
    with open('models\model.pkl', 'wb') as file:
        pickle.dump(classifier, file)

    with open('vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)


def load_dataset(data):
    reviews = []
    sentiments = []
    with open(data, encoding='utf-8') as data:
        reader = csv.reader(data)
        for row in reader:
            reviews.append(row[0])
            sentiment = 1 if row[1] == 'positive' else 0
            sentiments.append(sentiment)
    
    return (reviews, sentiments)

def negate_sentence(sentence):

    negation_markers = ["not", "no", "never", "neither", "without", "don't", "can't"]  # List of negation markers
    negated_sentence = []

    negate = False  # Flag to indicate if the current word should be negated

    for word in sentence:
        if word in negation_markers:
            negate = True  # Set the flag to True when a negation marker is encountered
        elif word in [".", "!", "?"]:
            negate = False  # Reset the flag at the end of a sentence or clause
        elif negate:
            word = "NOT_" + word  # Add a prefix to indicate negation
            negate = False  # Reset the flag after negating the current word
            continue  # Skip appending the negated word to exclude the subsequent word
        negated_sentence.append(word)

    return negated_sentence

def preprocess_data(reviews):
    
    #Initialize Lemmatizer
    lemmatizer = WordNetLemmatizer()

    preprocessed = []

    for review in reviews:
        if isinstance(review, str):
            # Convert words to lowercase and tokenize
            review = review.lower()

            review = word_tokenize(review)

            # Remove stopwords, they don't add much to our sentiment
            stopped_review = [word for word in review if word not in stopwords]

            # Lemmatize words to reduce dimensionality
            lemmatized = [lemmatizer.lemmatize(word) for word in stopped_review]

            # Handle very simple negations
            negated = negate_sentence(lemmatized)

            # Join the words back into a string
            preprocessed.append(' '.join(negated))



    return preprocessed



if __name__ == "__main__":
    main()
