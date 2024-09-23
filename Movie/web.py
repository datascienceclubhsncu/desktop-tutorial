import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download necessary NLTK data
nltk.download('stopwords')

# Define the genre mapping
genre_mapping = {'other': 0, 'action': 1, 'romance': 2, 'horror': 3, 'sci-fi': 4, 'comedy': 5, 'thriller': 6, 'drama': 7, 'adventure': 8}

# Load the dataset and preprocess
@st.cache
def load_data():
    df = pd.read_csv('/content/kaggle_movie_train.csv', engine='python', on_bad_lines='skip')
    df.drop(columns='id', inplace=True)
    df['genre'] = df['genre'].map(genre_mapping)
    return df

# Preprocess the text
def preprocess_text(text):
    ps = PorterStemmer()
    dialog = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text)
    dialog = dialog.lower()
    words = dialog.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Train the model
@st.cache(allow_output_mutation=True)
def train_model(df):
    corpus = [preprocess_text(text) for text in df['text']]
    
    cv = CountVectorizer(max_features=10000, ngram_range=(1, 2))
    X = cv.fit_transform(corpus).toarray()
    y = df['genre'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)
    
    y_pred = nb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return nb_classifier, cv, accuracy

# Function to predict genre
def predict_genre(text, model, vectorizer):
    processed_text = preprocess_text(text)
    text_vec = vectorizer.transform([processed_text])
    prediction = model.predict(text_vec)
    return prediction[0]

# Streamlit app layout
st.title("Movie Genre Predictor")

# Load data
df = load_data()

# Train the model
model, vectorizer, accuracy = train_model(df)

st.write(f"Model trained with an accuracy of {round(accuracy * 100, 2)}% on the test set")

# Input for movie description
movie_description = st.text_area("Enter the movie description:")

if st.button("Predict Genre"):
    if movie_description:
        genre = predict_genre(movie_description, model, vectorizer)
        genre_name = list(genre_mapping.keys())[list(genre_mapping.values()).index(genre)]
        st.write(f"The predicted genre is: **{genre_name}**")
    else:
        st.write("Please enter a movie description.")

