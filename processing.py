# File Imports
# from api.config import movies
import os
import numpy as np
import pandas as pd
import re
from dotenv import load_dotenv

# import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm
from transformers import pipeline
import nltk

# from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import emoji

# import tiktoken
from imdb import IMDb

import spacy

# from langdetect import detect

# import copy
import httplib2
from googleapiclient.discovery import build_from_document
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

# import time
# from openai.error import (
#     APIError,
#     OpenAIError,
#     RateLimitError,
#     ServiceUnavailableError,
#     Timeout,
#     TryAgain,
# )
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from bs4 import BeautifulSoup

# from google.colab import files
# from IPython.display import display, Markdown
# import io
# import openai
import html

# import json
# import ast
# import concurrent.futures
# import requests

nltk.download("stopwords")
nltk.download("vader_lexicon")
load_dotenv()
pd.options.display.max_colwidth = 100

classifier_1 = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    return_all_scores=True,
)
classifier_2 = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True,
)
tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 500}


def get_video_comments(service, **kwargs):
    comments = []
    results = service.commentThreads().list(**kwargs).execute()

    with tqdm() as progress_bar:  # create a progress bar
        while results:
            for item in results["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            progress_bar.update()  # update progress bar

            if "nextPageToken" in results:
                kwargs["pageToken"] = results["nextPageToken"]
                results = service.commentThreads().list(**kwargs).execute()
            else:
                break
    return comments


def generate_comments_df(video_id, key):
    # setup
    api_key = key
    http = httplib2.Http()
    service_name = "youtube"
    version = "v3"
    discovery_url = (
        f"https://www.googleapis.com/discovery/v1/apis/{service_name}/{version}/rest"
    )
    print("discovery_url", discovery_url)
    discovery_http = http.request(discovery_url)[1]
    youtube_service = build_from_document(discovery_http, developerKey=api_key)
    trailer_ids = [video_id]
    df_list = []
    print("trailer_ids", trailer_ids)
    for idx, video_id in enumerate(trailer_ids):
        comments = get_video_comments(youtube_service, part="snippet", videoId=video_id)
        df = pd.DataFrame(comments, columns=["comment"])
        df_list.append(df)
    temp = pd.concat(df_list)
    return temp


def remove_emojis_and_apostrophes(text):
    text = html.unescape(text)
    text = emoji.demojize(text)
    text = re.sub(r"<a\s+href=.*?>.*?</a>", "", text, flags=re.IGNORECASE)
    text = text.replace("'", "").replace('"', "")
    text = text.replace("“", "").replace("”", "")
    text = text.replace("‘", "").replace("’", "")
    text = (
        text.replace("<br>", "")
        .replace("<b>", "")
        .replace("<i>", "")
        .replace("</i>", "")
        .replace("</b>", "")
    )
    return text


def df_character_cleaning(df):
    df = df.iloc[:1000]
    temp_org = df.copy().reset_index(drop=True)
    comments_t = temp_org["comment"].tolist()
    comments_t = [remove_emojis_and_apostrophes(comment) for comment in comments_t]
    temp_org["comment"] = comments_t
    return temp_org


# Define a function to get classifier results for a single comment
def get_classifiers_output(comment):
    a = classifier_1(comment, **tokenizer_kwargs)
    b = classifier_2(comment, **tokenizer_kwargs)
    return a, b


def get_comments_sentiment(comments):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(get_classifiers_output, comments))
    all_scores = []
    for a, b in results:
        for l in range(3):
            all_scores.append(a[0][l]["score"])
        for m in range(6):
            all_scores.append(b[0][m]["score"])
    return all_scores


def comparison_table(all_scores, movie_id, movies):
    overall_sentiment = np.mean(np.array(all_scores).reshape(-1, 9), axis=0).tolist()
    print(overall_sentiment)
    overall_sentiment_df = pd.DataFrame(overall_sentiment)
    overall_sentiment_df = overall_sentiment_df.transpose()
    overall_sentiment_df.columns = [
        "negative",
        "neutral",
        "positive",
        "sadness",
        "joy",
        "love",
        "anger",
        "fear",
        "surprise",
    ]
    overall_sentiment_df = overall_sentiment_df.applymap(lambda x: f"{x*100:.1f}%")

    ia = IMDb()
    movie = ia.get_movie(movie_id)

    new_rows = pd.DataFrame(movies)
    titles = [
        movie.get("title"),
        "Barbie",
        "Guardians of the Galaxy Vol. 3",
        "Oppenheimer",
        "The Flash",
        "Mission Impossible: Dead Reckoning Part 1",
        "Spider-Man: Across the Spiderverse",
    ]
    overall_sentiment_df = pd.concat(
        [overall_sentiment_df, new_rows], axis=0
    ).reset_index(drop=True)
    overall_sentiment_df.insert(0, "Title", titles)
    print(overall_sentiment_df)
    return overall_sentiment_df


def get_movie_entities(movie_id):
    ia = IMDb()
    movie = ia.get_movie(movie_id)

    data = []
    # Entity Analysis

    def add_entities(entities, entity_type):
        for entity in entities:
            row = {
                "Name": entity["name"],
                "Notes": entity.notes if hasattr(entity, "notes") else "",
                "Entity Type": entity_type,
            }

            if (
                hasattr(entity.currentRole, "keys")
                and "name" in entity.currentRole.keys()
            ):
                for k, v in entity.currentRole.items():
                    row[f'currentRole["{k}"]'] = v

            elif len(entity.currentRole) > 1:
                for k, v in entity.currentRole[0].items():
                    row[f'currentRole["{k}"]'] = " / ".join(
                        [str(r[k]) for r in entity.currentRole]
                    )

            # Add any other available attributes as columns
            for key, value in entity.items():
                if key not in row:
                    row[key] = value

            data.append(row)

    entity_types = [
        "cast",
        "director",
        "writers",
        "producer",
        "cinematographer",
        "composer",
        "production companies",
    ]

    for entity_type in entity_types:
        # Edit to ensure all information is in the keys
        if entity_type in movie.keys():
            if entity_type == "production companies":
                for company in movie[entity_type]:
                    row = {
                        "Name": company.get("name", ""),
                        "Notes": company.get("notes", ""),
                        "Entity Type": "Production Company",
                        'currentRole["long imdb name"]': "Production Company",
                    }

                    # Add any other available attributes as columns
                    for key, value in company.items():
                        if key not in row:
                            row[key] = value

                    data.append(row)
            else:
                add_entities(movie[entity_type], entity_type.title())

    df = pd.DataFrame(data)
    return df


def create_entities_df(movie_id):
    df = get_movie_entities(movie_id)

    entities = (
        df[["Name", 'currentRole["long imdb name"]']]
        .fillna("")
        .rename(columns={'currentRole["long imdb name"]': "Role"})
    )

    return entities


# Basic entity match
def find_mentioned_entities(comments, entities):
    nlp = spacy.load("en_core_web_sm")

    mentioned_entities = []
    for comment in comments:
        doc = nlp(str(comment).lower())
        mentioned = [int(entity.lower() in doc.text) for entity in entities["Name"]]
        mentioned_entities.append(mentioned)

    mentioned_entities_df = pd.DataFrame(mentioned_entities, columns=entities["Name"])
    return mentioned_entities_df.T.drop_duplicates().T


# Function to analyze the sentiment of a comment and classify it as positive, neutral, or negative
def analyze_sentiment(comment):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(comment)

    # Classify the sentiment based on the compound score
    if sentiment_scores["compound"] >= 0.1:
        sentiment_class = "Positive"
    elif sentiment_scores["compound"] <= -0.1:
        sentiment_class = "Negative"
    else:
        sentiment_class = "Neutral"

    return sentiment_class


def sentiments_df(to_summarize):
    sentiments = []
    for comment in to_summarize:
        sentiment = analyze_sentiment(comment)
        sentiments.append(sentiment)
        sentiments_df = pd.DataFrame(sentiments, columns=["Sentiment"])
    return sentiments_df


def entities_table(entity_mentions, sentiments_df):
    mentions = entity_mentions.loc[:, (entity_mentions.sum(0) > 3)].copy()
    mentions = mentions.loc[mentions.sum(1) > 0, :].copy()
    sentiment_mentions = sentiments_df.loc[mentions.index, :].copy()
    mentions["sentiment"] = sentiment_mentions
    # Order the columns by total mentions
    column_order = (
        mentions[[c for c in mentions.columns if c != "sentiment"]]
        .sum(0)
        .sort_values(ascending=False)
        .index.tolist()
    )
    mentions_total = mentions[[c for c in mentions.columns if c != "sentiment"]].sum()
    # Calculate the sentiment breakdown for each person
    sentiment_breakdown = mentions.groupby("sentiment").sum().transpose()
    # Merge the data
    data = pd.concat([mentions_total, sentiment_breakdown], axis=1)
    # Rename the first column to "Total Mentions"
    data = data.rename(columns={0: "Total Mentions"})
    data = data.loc[column_order, :]
    return data.head(8)
