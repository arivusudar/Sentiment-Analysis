from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from flask_cors import CORS
# Download VADER lexicon (only first time)
nltk.download('vader_lexicon')

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    scores = sia.polarity_scores(text)

    sentiment = 'Neutral'
    if scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negative'

    return jsonify({
        'text': text,
        'scores': scores,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
