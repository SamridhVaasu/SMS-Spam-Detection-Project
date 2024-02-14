import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


# Define a function to preprocess the text
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [token for token in text if token.isalnum()]
    # Remove stopwords and punctuation
    text = [token for token in text if token not in stopwords.words('english') + list(string.punctuation)]
    # Stem the tokens
    text = [ps.stem(token) for token in text]
    # Join the tokens back to a string
    return " ".join(text)


# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set the page title and layout
st.set_page_config(page_title="Spam Classifier", layout="wide")

# Create a title and a subtitle
st.markdown("<h1 style='text-align: center;'>Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align: center;'>A simple app to classify messages as spam or not spam using natural language processing and machine learning</h3>",
    unsafe_allow_html=True)

# Create a sidebar with some information and instructions
st.sidebar.header("About")
st.sidebar.write(
    "This app uses a logistic regression model trained on a dataset of 5,572 messages to predict whether a given message is spam or not spam. The model achieves an accuracy of 98.2% on the test set. The app also uses NLTK to preprocess the text before feeding it to the model.")
st.sidebar.write(
    "To use the app, simply enter a message in the text area and click on the 'Predict' button. The app will display the prediction and the probability of the message being spam.")

# Create a text input box for user input
input_sms = st.text_area("Enter the message", height=200)

# Create a button to trigger the prediction
if st.button('Predict'):

    # Preprocess the input
    transformed_sms = transform_text(input_sms)
    # Vectorize the input
    vector_input = tfidf.transform([transformed_sms])
    # Make prediction
    result = model.predict(vector_input)[0]
    # Get the probability of the prediction
    proba = model.predict_proba(vector_input)[0][result]
    # Display the prediction and the probability
    if result == 1:
        st.error(f"This message is classified as Spam with {proba * 100:.2f}% probability.")
    else:
        st.success(f"This message is classified as Not Spam with {proba * 100:.2f}% probability.")
