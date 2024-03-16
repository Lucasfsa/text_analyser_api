from flask import Flask, request, jsonify
from core.core import get_sentiment, get_stats_sentiments

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def base_page():
  return jsonify('Welcome to text_analyser API'), 200


@app.route('/sentiment', methods=['POST'])
def sentment():
  data = request.get_json()
  return jsonify({'prediction':
                  get_sentiment(data['pre_processed_text'])}), 200


@app.route('/process-sentiments', methods=['POST'])
def process_sentiment():

  data = request.get_json()

  stats = get_stats_sentiments(data['pre_processed_text'])

  return jsonify({'prediction': stats}), 200


if __name__ == "__main__":
  app.run()
