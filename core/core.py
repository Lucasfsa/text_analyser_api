from core.knn import get_predict

def get_sentiment(text):
  return get_predict(text)


def get_stats_sentiments(pre_processed_text):
  stats = {
      "very_positive_predictions": 0,
      "very_negative_predictions": 0,
      "positive_predictions": 0,
      "negative_predictions": 0,
      "neutro_predictions": 0,
      'predicts': {}
  }

  texts = pre_processed_text.split("/")
  for text in texts:
    sentiment = get_predict(text)

    if sentiment == 'positivo':
      stats['positive_predictions'] += 1
    elif sentiment == 'muito positivo':
      stats['very_positive_predictions'] += 1
    elif sentiment == 'negativo':
      stats['negative_predictions'] += 1
    elif sentiment == 'muito negativo':
      stats['very_negative_predictions'] += 1
    else:
      stats['neutro_predictions'] += 1

    stats['predicts'][text] = sentiment

  return stats
