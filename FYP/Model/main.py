import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

sentiment_model = "FYP\Model\sentiment_model.pkl"
# Load the model
#with open('sentiment_model.pkl', 'rb') as file:
with open(sentiment_model, 'rb') as file:
    model = pickle.load(file)

# Assuming your dataset is in a CSV file called 'reviews.csv' with columns 'review' and 'rating'
data = pd.read_csv('FYP\Dataset\cleandata.csv')

# Separate the features (review text) and target (rating)
X = data['Review']
y = data['Rating']

# Convert the rating to binary sentiment labels (positive or negative)
y = y.apply(lambda x: 'positive' if x > 3 else 'negative')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

#review = "I love this product"


while(True):
    review = input("Enter Text: ")
    vectorized_review = vectorizer.transform([review])

    # Make predictions
    prediction = model.predict(vectorized_review)

    # Output the predicted sentiment
    print("Predicted sentiment: ", prediction[0])