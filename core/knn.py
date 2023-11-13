import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from googletrans import Translator
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean', algorithm='auto')
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
translator = Translator()


def get_sentiment(text):
  scores = sia.polarity_scores(text)
  if scores['compound'] > 0.6:
    return {'compound': scores['compound'], 'label': 'muito positivo'}
  elif scores['compound'] > 0.2 and scores['compound'] <= 0.6:
    return {'compound': scores['compound'], 'label': 'positivo'}
  elif scores['compound'] >= -0.2 and scores['compound'] <= 0.2:
    return {'compound': scores['compound'], 'label': 'neutro'}
  elif scores['compound'] >= -0.6 and scores['compound'] < -0.2:
    return {'compound': scores['compound'], 'label': 'negativo'}
  elif scores['compound'] < -0.6:
    return {'compound': scores['compound'], 'label': 'muito negativo'}


X = [[1.0], [0.9], [0.8], [0.7], [0.6], [0.5], [0.4], [0.3], [0.2], [0.1],
     [0.0], [-0.1], [-0.2], [-0.3], [-0.4], [-0.5], [-0.6], [-0.8], [-0.8],
     [-0.9], [-1.0]]

y = [
    'muito positivo', 'muito positivo', 'muito positivo', 'muito positivo',
    'positivo', 'positivo', 'positivo', 'positivo', 'neutro', 'neutro',
    'neutro', 'neutro', 'neutro', 'negativo', 'negativo', 'negativo',
    'negativo', 'muito negativo', 'muito negativo', 'muito negativo',
    'muito negativo'
]

knn.fit(X, y)


def get_predict(text):
  translated = translator.translate(text, dest='en')
  sentiment_scores = sia.polarity_scores(translated.text)
  data = [sentiment_scores['compound']]
  prediction = knn.predict([data])
  prediction_str = prediction[0]
  return prediction_str
