import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

data = pd.read_csv("spam.csv", encoding='latin-1')
# print(data.head(5))
# print(data.columns)
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# print(data.columns)
# print(data.head())
data['class'] = data['class'].map({'ham': 0, 'spam': 1})
# print(data.head())

cv = CountVectorizer()

x = data['message']
y = data['class']

# print(x.shape, y.shape)

x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = MultinomialNB()
model.fit(x_train, y_train)

result = model.score(x_test, y_test)
# print(result*100)

pickle.dump(model, open("spam.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))
clf = pickle.load(open("spam.pkl", "rb"))

# print(clf)
msg = input("message: ")
data = [msg]
vect = cv.transform(data).toarray()
result = model.predict(vect)
if result == 0:
    print(msg)
else:
    print("SPAM")
