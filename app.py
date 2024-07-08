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
    TOTAL_SUMMARIZER,
    create_movie_info,
)
from config import movies
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import openai


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
    url_pattern = r"(?:https?:\/\/)?(?:www\.)?imdb\.com\/title\/tt(\d+)"
    id_pattern = r"^(tt)?(\d+)$"
    url_match = re.search(url_pattern, input_string)
    if url_match:
        return url_match.group(1)
    id_match = re.match(id_pattern, input_string)
    if id_match:
        return id_match.group(2)
    else:
        return None


def log_progress(message, start_time):
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    time_str = f"{minutes:02d}:{seconds:02d}"
    st.sidebar.write(f"[{time_str}] {message}")


# def main():

#     st.set_page_config(page_title="Product Demo", layout="wide")
#     st.title("Product Demo")
#     col1, col2 = st.columns(2)
#     with col1:
#         youtube_ref = st.text_input("Youtube video id or link", key="video_id")
#         if youtube_ref:
#             video_id = extract_youtube_id(youtube_ref)
#             if video_id is None:
#                 st.error("Please enter a valid YouTube ref")
#     with col2:
#         imdb_ref = st.text_input("IMDB movie id or link", key="movie_id")
#         if imdb_ref:
#             movie_id = extract_imdb_id(imdb_ref)
#             if movie_id is None:
#                 st.error("Please enter a valid IMDB ref")

#     if col1.button("Submit"):

#         with st.spinner(
#             "Processing... (see progress in sidebar - average time 5-6mins)"
#         ):
#             start_time = time.time()
#             st.sidebar.markdown("**Progress**")

#             log_progress("Extracting comments...", start_time)
#             comments = generate_comments_df(video_id, st.secrets["YT_KEY"])
#             log_progress("Cleaning comments...", start_time)
#             comments = df_character_cleaning(comments)
#             log_progress("Calculating sentiment scores...", start_time)
#             all_scores = get_comments_sentiment(comments["comment"].tolist())
#             log_progress("Creating comparison table...", start_time)
#             temp = comparison_table(all_scores, movie_id, movies)
#             with col1:
#                 st.header("Movie Sentiment and Emotion Analysis")
#                 st.dataframe(temp.set_index("Title"))
#                 sentiment_viz(temp)

#             log_progress("Extracting entities...", start_time)
#             to_summarize = comments["comment"].astype(str).tolist()
#             entities = create_entities_df(movie_id)
#             log_progress("Finding entities...", start_time)
#             entity_mentions = find_mentioned_entities(to_summarize, entities)
#             sorted_values = entity_mentions.sum().sort_values(ascending=False).head(10)
#             log_progress("Matching sentiments to entities...", start_time)
#             sentiments_df = create_sentiments_df(to_summarize)
#             sorted_values_2 = sentiments_df["Sentiment"].value_counts()
#             temp = entities_table(entity_mentions, sentiments_df)
#             with col2:
#                 st.header("Entities Analysis")
#                 st.dataframe(temp.head(8))
#                 entity_viz(sorted_values, sorted_values_2, temp)
#             log_progress("Done!", start_time)

#     st.header("Advanced Topic Analysis")
#     openai_key = st.text_input(
#         "*Please enter openai api key for advanced topic analysis*", key="openai_key"
#     )
#     st.text_area("Summary", value="Openai analysis", height=300)


def main():
    st.set_page_config(page_title="Product Demo", layout="wide")
    st.title("Product Demo")

    # Initialize session state variables
    if "video_id" not in st.session_state:
        st.session_state.video_id = None
    if "movie_id" not in st.session_state:
        st.session_state.movie_id = None
    if "comments" not in st.session_state:
        st.session_state.comments = None
    if "all_scores" not in st.session_state:
        st.session_state.all_scores = None
    if "entities" not in st.session_state:
        st.session_state.entities = None
    if "entity_mentions" not in st.session_state:
        st.session_state.entity_mentions = None
    if "sentiments_df" not in st.session_state:
        st.session_state.sentiments_df = None
    if "sorted_values" not in st.session_state:
        st.session_state.sorted_values = None
    if "sorted_values_2" not in st.session_state:
        st.session_state.sorted_values_2 = None
    if "temp" not in st.session_state:
        st.session_state.temp = None
    if "temp_2" not in st.session_state:
        st.session_state.temp_2 = None
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = None
    if "movie_info_str" not in st.session_state:
        st.session_state.movie_info_str = None
    if "first_analysis_complete" not in st.session_state:
        st.session_state.first_analysis_complete = False
    if "table" not in st.session_state:
        st.session_state.table = None

    col1, col2 = st.columns(2)

    with col1:
        youtube_ref = st.text_input("Youtube video id or link")
        if youtube_ref:
            st.session_state.video_id = extract_youtube_id(youtube_ref)
            if st.session_state.video_id is None:
                st.error("Please enter a valid YouTube ref")

    with col2:
        imdb_ref = st.text_input("IMDB movie id or link")
        if imdb_ref:
            st.session_state.movie_id = extract_imdb_id(imdb_ref)
            if st.session_state.movie_id is None:
                st.error("Please enter a valid IMDB ref")

    st.sidebar.markdown("**Progress**")
    if col1.button("Submit"):
        with st.spinner(
            "Processing... (see progress in sidebar - average time 3-5mins)"
        ):
            start_time = time.time()
            log_progress("Extracting comments...", start_time)
            st.session_state.comments = generate_comments_df(
                st.session_state.video_id, st.secrets["YT_KEY"]
            )

            log_progress("Cleaning comments...", start_time)
            st.session_state.comments = df_character_cleaning(st.session_state.comments)

            log_progress("Calculating sentiment scores...", start_time)
            st.session_state.all_scores = get_comments_sentiment(
                st.session_state.comments["comment"].tolist()
            )

            log_progress("Creating comparison table...", start_time)
            st.session_state.table = comparison_table(
                st.session_state.all_scores, st.session_state.movie_id, movies
            )

            log_progress("Extracting entities...", start_time)
            to_summarize = st.session_state.comments["comment"].astype(str).tolist()
            st.session_state.entities, st.session_state.temp = create_entities_df(
                st.session_state.movie_id
            )
            # First part done

            log_progress("Finding entities...", start_time)
            st.session_state.entity_mentions = find_mentioned_entities(
                to_summarize, st.session_state.entities
            )
            st.session_state.sorted_values = (
                st.session_state.entity_mentions.sum()
                .sort_values(ascending=False)
                .head(10)
            )

            log_progress("Matching sentiments to entities...", start_time)
            st.session_state.sentiments_df = create_sentiments_df(to_summarize)
            st.session_state.sorted_values_2 = st.session_state.sentiments_df[
                "Sentiment"
            ].value_counts()
            st.session_state.temp_2 = entities_table(
                st.session_state.entity_mentions, st.session_state.sentiments_df
            )
            log_progress("Done!", start_time)
            st.session_state.first_analysis_complete = True

    if st.session_state.get("first_analysis_complete", False):
        with col1:
            st.header("Movie Sentiment and Emotion Analysis")
            st.dataframe(st.session_state.table.set_index("Title"))
            sentiment_viz(st.session_state.table)

        with col2:
            st.header("Entities Analysis")
            st.dataframe(st.session_state.temp_2.head(8))
            entity_viz(
                st.session_state.sorted_values,
                st.session_state.sorted_values_2,
                st.session_state.temp_2,
            )

    st.header("Advanced Topic Analysis")
    st.session_state.openai_key = st.text_input(
        "*Please enter openai api key for advanced topic analysis*"
    )
    col1, col2 = st.columns(2)
    if col1.button("Send"):
        openai.api_key = st.session_state.openai_key
        st.session_state.movie_info_str = create_movie_info(
            st.session_state.movie_id, st.session_state.temp
        )
        display_summary(st.session_state.comments, st.session_state.movie_info_str)


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


def display_summary(df, movie_info_str):
    temp_org = df.copy().reset_index(drop=True)
    to_summarize = temp_org["comment"].astype(str).tolist()
    all_resp = TOTAL_SUMMARIZER(to_summarize, 3900, movie_info_str)
    resp_list = [item for item in all_resp.splitlines() if item]
    print(resp_list)
    st.markdown("### Summarized Output")
    st.markdown(
        "Below is the output from chunking the text and recursively querying OpenAI to summarize the comments:"
    )

    st.markdown("### Aspects of the trailer/film that commenters like:")
    likes = "".join([f"- {item}\n" for item in resp_list[:5]])
    st.markdown(likes)

    st.markdown("### Aspects of the trailer/film that commenters dislike:")
    dislikes = "".join([f"- {item}\n" for item in resp_list[5:10]])
    st.markdown(dislikes)


if __name__ == "__main__":
    main()
