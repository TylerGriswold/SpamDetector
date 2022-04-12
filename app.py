import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup


ps = PorterStemmer()

def data_cleanse(email):
    email = email.lower()
    email = nltk.word_tokenize(email)

    cleansedEmail = []
    for i in email:
        if i.isalnum():
            cleansedEmail.append(i)

    email = cleansedEmail[:]
    cleansedEmail.clear()

    for i in email:
        if i not in stopwords.words('english') and i not in string.punctuation:
            cleansedEmail.append(i)
    email = cleansedEmail[:]
    cleansedEmail.clear()

    for i in email:
        cleansedEmail.append(ps.stem(i))

    return " ".join(cleansedEmail)

from nylas import APIClient
CLIENT_ID = "2vcrdrty1sud9djc6rjlgipxd"
CLIENT_SECRET = "8hwehj7h84noctkjqmpsvspvq"
ACCESS_TOKEN = "jjAJFaGqjvIQU5C4RPueVXc3jNjq8f"
nylas = APIClient(
    CLIENT_ID,
    CLIENT_SECRET,
    ACCESS_TOKEN
)

message = nylas.messages.first()
word = ""
word = ("from: {} | to: {} | ID: {}| Body: {}".format(
    message.from_, message.to, message.id, message.body
))
#word.replace('\n', ' ')
#word.replace('\t', ' ')
clean = BeautifulSoup(word, 'lxml').text

tfidf = pickle.load(open('tfidf.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Detector")
st.write('The email from :', message.from_[0]["name"], message.from_[0]["email"])
st.write('SUBJECT: ', message.subject)
st.write('Brief Message: ', message.snippet)

#perform spam check
email = data_cleanse(clean)
vector_input = tfidf.transform([email])
result = model.predict(vector_input)[0]
#display result
if result == 1:
    st.header("THE LAST EMAIL RECEIVED IS SPAM")
else:
    st.header("THE LAST EMAIL RECEIVED IS NOT SPAM")

#sidebar code begins
st.sidebar.title('Check For Spam')
emailMessage = st.sidebar.text_area('Enter email to check for spam:')
#when button is pressed, user inputed email is checked
if st.sidebar.button('Predict'):
    email = data_cleanse(emailMessage)
    vector_input = tfidf.transform([email])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.sidebar.header("IS SPAM")
    else:
        st.sidebar.header("IS NOT SPAM")

unsafe_allow_html= True
# https://developer.nylas.com/docs/user-experience/components/mailbox-component/#quickstart
mailbox = '<html><head><script type="text/javascript" src="dist/purify.min.js"></script><script src="https://unpkg.com/@nylas/components-mailbox"></script></head><body><nylas-mailbox id="8a960af4-0b11-468c-a08a-49fe45f112f4"></nylas-mailbox></body></html>'
st.components.v1.html(mailbox)

