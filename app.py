"""CS4106: sentiment classification with honest handling of uncertainty."""
import logging
import streamlit as st
from transformers import pipeline

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"

st.set_page_config(page_title="Sentiment Studio", page_icon="💬", layout="centered")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model=MODEL_ID,
                    revision=MODEL_REVISION, device=-1)

def normalize(text):
    """Use the model author's recommended mention and URL normalization."""
    return " ".join("@user" if word.startswith("@") and len(word) > 1
                    else "http" if word.startswith("http") else word
                    for word in text.split())

def summarize(scores):
    ordered = sorted(scores, key=lambda item: item["score"], reverse=True)
    top, second = ordered[:2]
    # An application display rule, not a calibrated statistical threshold.
    uncertain = top["score"] < 0.65 or top["score"] - second["score"] < 0.20
    return "UNCLEAR" if uncertain else top["label"].upper(), ordered

st.caption("CS4106 · SESSION 5 · STREAMLIT + HUGGING FACE")
st.title("💬 Sentiment Studio")
st.write("Explore the tone of an English message: positive, neutral or negative.")
with st.sidebar:
    st.header("About this app")
    st.write("RoBERTa recognizes **positive**, **neutral** and **negative** sentiment.")
    st.caption("Trained for social-media language. Uncertain predictions are shown as UNCLEAR.")
    st.markdown(f"[View the Hugging Face model](https://huggingface.co/{MODEL_ID})")
    st.info("A model score is not the probability that an interpretation is correct. Tone, sarcasm and relationship context can change the meaning.")
    st.caption("Inference runs on the machine hosting this app. The first analysis downloads the model.")

with st.form("sentiment_form"):
    text = st.text_area("Your text", height=170, max_chars=20000,
                        placeholder="The service was excellent and the staff were so helpful!")
    submitted = st.form_submit_button("Analyze sentiment", type="primary", use_container_width=True)

if submitted:
    st.session_state.pop("analysis", None)
    clean_text = text.strip()
    if not clean_text:
        st.warning("Please enter a sentence before analyzing.")
    elif not any(char.isalpha() for char in clean_text):
        st.warning("Please include words. Emojis or punctuation alone do not provide enough context for this app.")
    else:
        try:
            with st.spinner("Reading your text… First use may take a few minutes."):
                classifier = load_model()
                processed = normalize(clean_text)
                token_count = len(classifier.tokenizer.encode(processed, add_special_tokens=True, truncation=False, verbose=False))
                if token_count > 512:
                    st.warning("Please shorten your text: it exceeds 512 model tokens. No prediction was made, so the end of your message is not silently ignored.")
                else:
                    scores = classifier(processed, top_k=None, truncation=False)
                    label, ordered = summarize(scores)
                    st.session_state.analysis = {"text": clean_text, "label": label, "scores": ordered}
        except Exception:
            logging.exception("Sentiment analysis failed")
            st.error("The model could not run. Check the internet connection and available memory, then try again. On Hugging Face Spaces, check the runtime logs.")

if "analysis" in st.session_state:
    result = st.session_state.analysis
    st.divider()
    st.caption("Result for: " + result["text"])
    left, right = st.columns(2)
    left.metric("Predicted sentiment", result["label"])
    right.metric("Top model score", f"{result['scores'][0]['score']:.1%}")
    if result["label"] == "UNCLEAR":
        st.info("The scores do not strongly favor one interpretation. Add context or treat this message as ambiguous.")
    for item in result["scores"]:
        st.progress(float(item["score"]), text=f"{item['label'].capitalize()}: {item['score']:.1%}")
    st.caption("These scores describe the model's preference among labels, not your intent. Context-dependent messages can still be misclassified.")
    report = f"Text: {result['text']}\nDisplayed sentiment: {result['label']}\n"
    report += "\n".join(f"{item['label']}: {item['score']:.1%}" for item in result["scores"])
    report += f"\nModel: {MODEL_ID}\nRevision: {MODEL_REVISION}\n"
    st.download_button("Download result", report, file_name="sentiment-result.txt", mime="text/plain")

with st.expander("How to interpret the result"):
    st.write("Neutral is a learned model category. UNCLEAR is an app rule: the top score is below 65%, or the gap to the runner-up is below 20 percentage points. These thresholds are heuristics, not validated accuracy guarantees.")
    st.write("Try a positive review, a complaint, a factual statement and a sarcastic sentence. For example: 'I think we should be more than friends' needs relationship context; no single sentence can establish the speaker's intent.")

st.caption("Built for learning · English text · Review predictions before using them")
