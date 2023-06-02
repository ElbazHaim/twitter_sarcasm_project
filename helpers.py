""" 
Helper module for the twitter sarcasm project, holds processing functions imported
by the project's jupyter notebook. 
"""

import re
import nltk
import pandas as pd 
from nltk import pos_tag
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def _remove_url(text: str) -> str:
    text=str(text)
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def _remove_emoji(text: str) -> str:
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def _remove_mentions(text: str) -> str:
    ment = re.compile(r"(@[A-Za-z0-9]+)")
    return ment.sub(r'', text)


def _remove_html(text: str) -> str:
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)


def _remove_iron_sarc(string:str) -> str:
    patterns = [r"\S*#(?:\[[^\]]+\]|\S+)", r'\b\w*iron\w*\b']    
    for pattern in patterns:
        string = re.sub(pattern, '', string)
    return string


def _remove_hashtag(string: str) -> str:
    return re.sub(r'\b\w*sarc\w*\b', '', string)
    

def clean_tweet(text: str) -> str:
    text = text.lower()
    text = _remove_url(text)
    text = _remove_html(text)
    text = _remove_emoji(text)
    text = _remove_mentions(text)
    text = _remove_iron_sarc(text)
    text = _remove_hashtag(text)
    text = ' '.join(nltk.word_tokenize(text)).lower()
    return text


def plot_wordcloud(dataset: pd.DataFrame, class_to_plot: str) -> None:
    stopwords = nltk.corpus.stopwords.words('english')
    plt.figure(figsize=(12,6))
    text = ' '.join(dataset.tweets[dataset["class"]==class_to_plot]).lower()
    wc = WordCloud(width=800, height=400, background_color='black').generate(text)
    plt.imshow(wc)
    

def encoder(t_class:str) -> int:
    t_class=str(t_class)
    class_dict = {
        'regular':0,
        'sarcasm_irony':1,
    }
    return class_dict[t_class]


def preprocessing_pipeline(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df = df.drop(df[df['class'] == "figurative"].index)
    df = df.drop_duplicates(subset=["tweets"])
    df.replace(["sarcasm", "irony"], "sarcasm_irony", inplace=True)
    df["tweets"] = df["tweets"].apply(clean_tweet)
    df["class"] = df['class'].apply(lambda x: encoder(x))
    return df


def count_syntactic_features(text: str) -> dict:
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    stop_words = set(stopwords.words('english'))
    counts = {
        'Stopwords': 0,
        'Nouns': 0,
        'Verbs': 0,
        'Adverbs': 0,
        'Adjectives': 0,
        'Pronouns': 0
    }
    for word, pos in tagged_tokens:
        word_lower = word.lower()
        if word_lower in stop_words:
            counts['Stopwords'] += 1
        elif pos.startswith(('NN', 'NNS', 'NNP', 'NNPS')):
            counts['Nouns'] += 1
        elif pos.startswith('VB'):
            counts['Verbs'] += 1
        elif pos.startswith('RB'):
            counts['Adverbs'] += 1
        elif pos.startswith('JJ'):
            counts['Adjectives'] += 1
        elif pos.startswith('PRP'):
            counts['Pronouns'] += 1
    return counts
