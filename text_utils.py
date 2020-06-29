from nltk.data import find
from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize.regexp import regexp_tokenize
from string import punctuation
from unidecode import unidecode
import re

## Download stopwords
try:
    find("corpora/stopwords")
except LookupError:
    download('stopwords')

## Regex to remove stopwords
STOPWORDS = ["pra", "pras"] + stopwords.words(["english", "portuguese"])
STOPWORDS = unidecode(' '.join(STOPWORDS)).lower().split()
STOPWORDS = list(set(STOPWORDS))
STOPWORDS_RE = re.compile(r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*')

## Regex to remove punctuations
PUNCTS_RE = re.compile('[' + punctuation + ']')

## Regex to remove punctutations and numbers
NON_ALPHANUM_RE = re.compile('[^a-zA-Z]')

## Regex to remove single character words
SINGLE_CHAR_RE = re.compile(r'\s+[a-zA-Z]\s+')

## Regex to remove multiple spaces
MULTI_SPACE_RE = re.compile(r'\s+')

def clean_text_min(text):
    """
        text: text to be cleaned
        output: formatted text

        reference: Tomas Mikolov, word2vec-toolkit: google groups thread., 2015. [](https://goo.gl/KtDGst)
    """

    # Decode to ascii
    text = unidecode(text.lower())

    # Remove punctuations
    text = PUNCTS_RE.sub(' ', text)
    #text = NON_ALPHANUM_RE.sub(' ', text)

    # Remove multiple spaces
    text = MULTI_SPACE_RE.sub(' ', text)

    return text.strip()


def clean_text(text):
    """
        text: text to be cleaned
        output: formatted text
    """

    # Decode to ascii
    text = unidecode(text.lower())
    
    # Remove punctuations and numbers
    text = NON_ALPHANUM_RE.sub(' ', text)

    # Remove single chars 
    text = SINGLE_CHAR_RE.sub(' ', text)

    # Remove stopwords
    text = STOPWORDS_RE.sub(' ', text)

    # Remove multiple spaces
    text = MULTI_SPACE_RE.sub(' ', text)

    return text.strip()

def tokenize_text(text, clean=False):
    """
        text: text to tokenize
        output: list of words
    """

    if clean:
        text = clean_text(text)

    words = regexp_tokenize(text, pattern="\s+", gaps=True)

    return(words)
