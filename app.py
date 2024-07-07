import streamlit as st
import time
from processing import (
    generate_comments_df,
    df_character_cleaning,
    get_comments_sentiment,
    comparison_table,
    create_entities_df,
    find_mentioned_entities,
    create_sentiments_df,
    entities_table,
)
from config import movies
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re


def extract_youtube_id(input_string):
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?(?:embed\/)?(?:v\/)?(?:shorts\/)?(?P<id>[^\s&?\/]+)"
    match = re.search(pattern, input_string)
    if match:
        return match.group("id")
    elif re.match(r"^[a-zA-Z0-9_-]{11}$", input_string):
        return input_string
    else:
        return None


def extract_imdb_id(input_string):
    pattern = r"(?:https?:\/\/)?(?:www\.)?imdb\.com\/title\/tt(\d+)"
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)
    elif re.match(r"^tt?\d+$", input_string):
        return input_string.lstrip("tt")
    else:
        return None


def log_progress(message, start_time):
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    time_str = f"{minutes:02d}:{seconds:02d}"
    st.sidebar.write(f"[{time_str}] {message}")


def main():

    st.set_page_config(page_title="Product Demo", layout="wide")
    st.title("Product Demo")
    col1, col2 = st.columns(2)
    with col1:
        youtube_ref = st.text_input("Youtube video id or link", key="video_id")
        if youtube_ref:
            video_id = extract_youtube_id(youtube_ref)
            if video_id is None:
                st.error("Please enter a valid YouTube ref")
    with col2:
        imdb_ref = st.text_input("IMDB movie id or link", key="movie_id")
        if imdb_ref:
            movie_id = extract_imdb_id(imdb_ref)
            if movie_id is None:
                st.error("Please enter a valid IMDB ref")

    if col1.button("Submit"):

        with st.spinner(
            "Processing... (see progress in sidebar - average time 5-6mins)"
        ):
            start_time = time.time()
            st.sidebar.markdown("**Progress**")

            log_progress("Extracting comments...", start_time)
            comments = generate_comments_df(video_id, st.secrets["YT_KEY"])
            log_progress("Cleaning comments...", start_time)
            comments = df_character_cleaning(comments)
            log_progress("Calculating sentiment scores...", start_time)
            all_scores = get_comments_sentiment(comments["comment"].tolist())
            log_progress("Creating comparison table...", start_time)
            temp = comparison_table(all_scores, movie_id, movies)
            with col1:
                st.header("Movie Sentiment and Emotion Analysis")
                st.dataframe(temp.set_index("Title"))
                sentiment_viz(temp)

            log_progress("Extracting entities...", start_time)
            to_summarize = comments["comment"].astype(str).tolist()
            entities = create_entities_df(movie_id)
            log_progress("Finding entities...", start_time)
            entity_mentions = find_mentioned_entities(to_summarize, entities)
            sorted_values = entity_mentions.sum().sort_values(ascending=False).head(10)
            log_progress("Matching sentiments to entities...", start_time)
            sentiments_df = create_sentiments_df(to_summarize)
            sorted_values_2 = sentiments_df["Sentiment"].value_counts()
            temp = entities_table(entity_mentions, sentiments_df)
            with col2:
                st.header("Entities Analysis")
                st.dataframe(temp.head(8))
                entity_viz(sorted_values, sorted_values_2, temp)
            log_progress("Done!", start_time)

    st.header("Advanced Topic Analysis")
    openai_key = st.text_input(
        "*Please enter openai api key for advanced topic analysis*", key="openai_key"
    )
    st.text_area("Summary", value="Openai analysis", height=300)


def percent_to_float(s):
    return float(s.strip("%")) / 100


def sentiment_viz(overall_sentiment_df):
    # Pre-processing
    sentiment_df = overall_sentiment_df[
        ["Title", "negative", "neutral", "positive"]
    ].copy()
    emotion_df = overall_sentiment_df[
        ["Title", "sadness", "joy", "love", "anger", "fear", "surprise"]
    ].copy()

    for col in sentiment_df.columns[1:]:
        sentiment_df[col] = sentiment_df[col].apply(percent_to_float)

    for col in emotion_df.columns[1:]:
        emotion_df[col] = emotion_df[col].apply(percent_to_float)

    # Sentiment Analysis
    st.subheader("Sentiment Analysis")
    fig, ax = plt.subplots(figsize=(18, 10))
    n_movies = len(sentiment_df["Title"])
    width = 0.12
    spacing = 0.02
    colors = plt.cm.viridis(np.linspace(0, 1, n_movies))
    for idx, movie in enumerate(sentiment_df["Title"]):
        plt.bar(
            [p + idx * (width + spacing) for p in range(len(sentiment_df.columns[1:]))],
            sentiment_df.loc[idx, sentiment_df.columns[1:]],
            width=width,
            label=movie,
            color=colors[idx],
        )
    ax.set_xticks(
        [
            p + (width + spacing) * n_movies / 2
            for p in range(len(sentiment_df.columns[1:]))
        ]
    )
    ax.set_xticklabels(sentiment_df.columns[1:], fontsize=14)
    legend = plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, fontsize=12)
    legend.set_title("Movies", prop={"size": 14})  # Set legend title
    legend.get_frame().set_facecolor("white")  # Set legend background color
    legend.get_frame().set_edgecolor("black")
    plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment="right")
    ax.set_title("Sentiment Analysis", fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel("Sentiments", fontsize=16)
    ax.set_ylabel("Percentage", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    # Emotion Analysis
    st.subheader("Emotion Analysis")
    fig, ax = plt.subplots(figsize=(18, 10))
    n_movies = len(emotion_df["Title"])
    width = 0.12
    spacing = 0.02
    colors = plt.cm.viridis(np.linspace(0, 1, n_movies))
    for idx, movie in enumerate(emotion_df["Title"]):
        plt.bar(
            [p + idx * (width + spacing) for p in range(len(emotion_df.columns[1:]))],
            emotion_df.loc[idx, emotion_df.columns[1:]],
            width=width,
            label=movie,
            color=colors[idx],
        )
    ax.set_xticks(
        [
            p + (width + spacing) * n_movies / 2
            for p in range(len(emotion_df.columns[1:]))
        ]
    )
    ax.set_xticklabels(emotion_df.columns[1:], fontsize=14)
    legend = plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, fontsize=12)
    legend.set_title("Movies", prop={"size": 14})
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment="right")
    ax.set_title("Emotion Analysis", fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel("Emotions", fontsize=16)
    ax.set_ylabel("Percentage", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)


def entity_viz(sorted_values, sorted_values_2, data):
    # Custom color palette: muted green for positive, and two other colors from coolwarm for neutral and negative
    muted_green = "#6dbf67"
    colors = sns.diverging_palette(250, 10, s=75, l=55, n=2)
    coolwarm_blue = colors[0]
    coolwarm_red = colors[1]
    palette = [muted_green, coolwarm_blue, coolwarm_red]

    # 1st chart: Bar chart for sorted_values
    fig, ax = plt.subplots(figsize=(18, 8))
    st.subheader("Entity mentions")
    sns.barplot(
        x=sorted_values.index,
        y=sorted_values,
        ax=ax,
        color=coolwarm_blue,
    )
    ax.set_title("Top Mentioned Entities", fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel("")
    ax.set_ylabel("Counts", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    # 2nd chart: Bar chart for top mentioned entities
    st.subheader("Global Entity Sentiments")
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.barplot(
        x=sorted_values_2.index,
        y=sorted_values_2,
        ax=ax,
        palette=palette,
        order=["Positive", "Neutral", "Negative"],
    )
    ax.set_title("Sentiment Distribution", fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel("")
    ax.set_ylabel("Counts", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    # 3rd chart: Sentiment breakdown bar chart
    st.subheader("Individual Entity Sentiments")
    fig, ax = plt.subplots(figsize=(18, 8))
    data[["Positive", "Neutral", "Negative"]].plot(
        kind="bar", stacked=True, ax=ax, color=palette
    )
    ax.set_title(
        "Sentiment Analysis of Comments Mentioning Different People",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Sentiments", fontsize=16)
    ax.legend(
        fontsize=14,
        loc="upper right",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
