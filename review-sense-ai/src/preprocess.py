import re
import string
import nltk
from nltk.corpus import stopwords
def get_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        return set(stopwords.words("english"))

STOP_WORDS = get_stopwords()
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)
