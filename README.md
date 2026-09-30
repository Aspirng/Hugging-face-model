# Sentiment Studio

Streamlit sentiment app using [Cardiff NLP's Twitter RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), licensed CC BY 4.0. Model weights are unmodified and pinned to revision `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`.

Outputs positive, neutral or negative scores. The app displays UNCLEAR when the top score is below 65% or the gap to second place is below 20 percentage points. These are display heuristics, not accuracy guarantees. Sarcasm and missing context remain limitations.

Run with `pip install -r requirements.txt` then `streamlit run app.py`. Model downloads on first use. Intended for learning.
