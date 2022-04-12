import string

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('D:/GRADSCHOOL/Capstone/emails/hamnspam/ham'):
    for filename in filenames:
        os.path.join(dirname, filename)

print("Size of Spam Data:", len(os.listdir('D:/GRADSCHOOL/Capstone/emails/hamnspam/spam')))
print("Size of Ham Data:", len(os.listdir('D:/GRADSCHOOL/Capstone/emails/hamnspam/ham')))

path = 'D:/GRADSCHOOL/Capstone/emails/hamnspam/'
mails = []
labels = []

for label in ['ham/', 'spam/']:
#     filenames = os.listdir(os.path.join(path, label))
    filenames = os.listdir(path + label)
    for file in filenames:
        f = open((path + label + file), 'r', encoding = 'latin-1')
        bolk = f.read()
        mails.append(bolk)
        labels.append(label)

df = pd.DataFrame({'emails': mails, 'labels': labels})
print(df.head(5))

#1

#changes label from ham/spam to 0/1


#print(df.isnull().sum())
#df['emails'] = df['emails'].apply(lambda x: x.lower())
#df['emails'] = df['emails'].apply(lambda x: x.replace('\n', ' '))
#df['emails'] = df['emails'].apply(lambda x: x.replace('\t', ' '))


import nltk
import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#nltk.download('stopwords')
stopwords.words('english')
#nltk. download('punkt')
ps = PorterStemmer()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['labels'] = encoder.fit_transform(df['labels'])


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

print(data_cleanse("from: [{'email': 'tjgriswold98@gmail.com', 'name': 'Tyler Griswold'}] | to: [{'email': 'griswoldtj17@uww.edu', 'name': ''}] | ID: cu1m2foql0dkefuyxlg9befmd| Body: EXTERNAL EMAILHi Tris, Tyler, Abraham and RuhongekaThe date of the capstone presentation is May 6 friday starting at 10:00am. Each of you will have 25 mins for the presentation + 5 additional time for questions. We would appreciate if you can be physically present but if this is not possible, we can schedule it over webex as well.The order will be 10:00 am Abraham10:30am Ruhongeka11am: Tris11:30am Tyler Please check with your project advisor if they are available during this time. Also please send Sue an abstract of your talk so she can help advertise itthanksLopa"))

df['cleansed_emails']= df['emails'].apply(data_cleanse)

print(df.head)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
#max features better results
tfidf = TfidfVectorizer(max_features = 2500)
review = tfidf.fit_transform(df['cleansed_emails']).toarray()
labels = df['labels'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(review, labels, test_size=0.2, random_state=2)

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
mnb = MultinomialNB()
bnb = BernoulliNB()
gnb = GaussianNB()

mnb.fit(X_train, y_train)
y_predict1 = mnb.predict(X_test)
print("mnb results")
print(accuracy_score(y_test, y_predict1))
print(confusion_matrix(y_test, y_predict1))
print(precision_score(y_test, y_predict1))
print()

bnb.fit(X_train, y_train)
y_predict2 = bnb.predict(X_test)
print("bnb results")
print(accuracy_score(y_test, y_predict2))
print(confusion_matrix(y_test, y_predict2))
print(precision_score(y_test, y_predict2))
print()

gnb.fit(X_train, y_train)
y_predict3 = gnb.predict(X_test)
print("gnb results")
print(accuracy_score(y_test, y_predict3))
print(confusion_matrix(y_test, y_predict3))
print(precision_score(y_test, y_predict3))
print()

import pickle

pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))

