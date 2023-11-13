import random, string
from flask import Flask, render_template, request, jsonify
from core.core import get_sentiment, get_stats_sentiments
from flask_cors import CORS

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)


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
  app.run(host='0.0.0.0', port=random.randint(2000, 9000))
