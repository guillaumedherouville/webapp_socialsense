import streamlit as st
import time
from processing import (
    generate_comments_df,
    df_character_cleaning,
    get_comments_sentiment,
    comparison_table,
    create_entities_df,
    find_mentioned_entities,
    sentiment_by_comment,
    entities_table,
    TOTAL_SUMMARIZER,
    create_movie_info,
    generate_summary_marketing,
    step_1,
    step_2,
    step_3,
)
from config import movies
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import openai
import pandas as pd


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


def main():
    st.set_page_config(page_title="SocialSense by Jumpcut", layout="wide")
    st.markdown(
        """
        <style>
        .center-text {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="center-text"><h1>{}</h1></div>'.format("SocialSense by Jumpcut"),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="center-text"><h2>{}</h2></div>'.format(
            "Social Intelligence and Marketing Optimization"
        ),
        unsafe_allow_html=True,
    )

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
    if "resp_list" not in st.session_state:
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
            st.session_state.sentiments_df = sentiment_by_comment(to_summarize)
            st.session_state.sorted_values_2 = st.session_state.sentiments_df[
                "Sentiment"
            ].value_counts()
            st.session_state.temp_2 = entities_table(
                st.session_state.entity_mentions, st.session_state.sentiments_df
            )
            log_progress("Done!", start_time)
            st.session_state.first_analysis_complete = True

    if st.session_state.get("first_analysis_complete", False):
        st.header("Movie Sentiment and Emotion Analysis")
        st.subheader("Sentiment Analysis")
        col1, col2 = st.columns(2)
        # col1.markdown(
        #     '<p style="font-size:14px; font-weight:bold;">Comparison of Sentiment Scores with famous movies</p>',
        #     unsafe_allow_html=True,
        # )
        col1.text("")
        col1.dataframe(
            st.session_state.table.set_index("Title")[
                ["negative", "neutral", "positive"]
            ]
        )
        with col2:
            sentiment_viz(st.session_state.table)
        st.subheader("Emotion Analysis")
        col1, col2 = st.columns(2)
        # col1.markdown(
        #     '<p style="font-size:14px; font-weight:bold;">Comparison of Emotion Scores with famous movies</p>',
        #     unsafe_allow_html=True,
        # )
        col1.text("")
        col1.dataframe(
            st.session_state.table.set_index("Title")[
                ["sadness", "joy", "love", "anger", "fear", "surprise"]
            ]
        )
        with col2:
            emotion_viz(st.session_state.table)

        st.header("Most Discussed People & Organisations")
        entity_viz(
            st.session_state.sorted_values,
            st.session_state.sorted_values_2,
            st.session_state.temp_2,
        )

        st.header("Advanced Topic Analysis")
        openai.api_key = st.secrets["OPENAI"]
        st.session_state.movie_info_str = create_movie_info(
            st.session_state.movie_id, st.session_state.temp
        )
        st.session_state.resp_list, st.session_state.temp, st.session_state.temp_2 = (
            summarize_comments(
                st.session_state.comments, st.session_state.movie_info_str
            )
        )
        display_summary(st.session_state.resp_list)
        # st.subheader("Number of Comments by Topic")
        # st.session_state.temp = step_1(
        #     st.session_state.temp, all_resp=st.session_state.temp_2
        # )
        # st.session_state.temp = step_2(st.session_state.temp)
        # print(st.session_state.temp)
        # st.session_state.temp = step_3(st.session_state.temp)
        # print(st.session_state.temp)
        # viz_comments_by_topic(st.session_state.temp)
        st.subheader("Marketing Actions")
        display_marketing(st.session_state.resp_list, st.session_state.movie_info_str)


def percent_to_float(s):
    return float(s.strip("%")) / 100


def sentiment_viz(overall_sentiment_df):
    # Pre-processing
    sentiment_df = overall_sentiment_df[
        ["Title", "negative", "neutral", "positive"]
    ].copy()
    for col in sentiment_df.columns[1:]:
        sentiment_df[col] = sentiment_df[col].apply(percent_to_float)

    # Sentiment Vizualization
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
    ax.set_title(
        "Comparison of Sentiment for Various Movies",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Sentiments", fontsize=16)
    ax.set_ylabel("Percentage", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)


def emotion_viz(overall_sentiment_df):
    # Pre-processing
    emotion_df = overall_sentiment_df[
        ["Title", "sadness", "joy", "love", "anger", "fear", "surprise"]
    ].copy()
    for col in emotion_df.columns[1:]:
        emotion_df[col] = emotion_df[col].apply(percent_to_float)

    # Emotion Vizualization
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
    ax.set_title(
        "Comparison of Emotion for Various Movies",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
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
    st.subheader("Top mentions")
    sns.barplot(
        x=sorted_values.index,
        y=sorted_values,
        ax=ax,
        color=coolwarm_blue,
    )
    ax.set_title("Top Mentioned People & Orgs", fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    # 2nd chart: Sentiment breakdown bar chart
    st.subheader("Individual Sentiments")
    fig, ax = plt.subplots(figsize=(18, 8))
    data[["Positive", "Neutral", "Negative"]].plot(
        kind="bar", stacked=True, ax=ax, color=palette
    )
    ax.set_title(
        "Sentiment Analysis of Comments Mentioning Different People & Orgs",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=16)
    ax.legend(
        fontsize=14,
        loc="upper right",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


def summarize_comments(df, movie_info_str):
    temp_org = df.copy().reset_index(drop=True)
    to_summarize = temp_org["comment"].astype(str).tolist()
    all_resp = TOTAL_SUMMARIZER(to_summarize, 3900, movie_info_str)
    resp_list = [item for item in all_resp.splitlines() if item]
    return resp_list, to_summarize, all_resp


def display_summary(resp_list):
    st.markdown("#### Aspects of the trailer/film that commenters like:")
    likes = "".join([f"{item}\n" for item in resp_list[:5]])
    st.markdown(likes)

    st.markdown("#### Aspects of the trailer/film that commenters dislike:")
    dislikes = "".join([f"{item}\n" for item in resp_list[5:10]])
    st.markdown(dislikes)


def display_marketing(resp_list, movie_info_str):
    marketing_actions = generate_summary_marketing(resp_list, movie_info_str)
    resp_list_mark = marketing_actions.splitlines()

    # Display results
    st.markdown("\n".join(resp_list_mark))


# def viz_comments_by_topic(new_data):
#     fig, ax = plt.subplots(figsize=(18, 8))
#     df = pd.DataFrame(new_data)
#     print("First few rows of new_data:", df.head(10))
#     df_ones = df.drop([0], inplace=False)
#     df_melted = df.melt(
#         value_vars=df_ones.columns, var_name="Columns", value_name="Count"
#     )

#     # Filter out rows with zeros
#     # df_ones = df_melted[df_melted["Count"] == 1]

#     # Define your color palette
#     muted_green = "#6dbf67"
#     colors = sns.diverging_palette(250, 10, s=75, l=55, n=2)
#     coolwarm_red = colors[1]
#     palette = [muted_green] * 5 + [
#         coolwarm_red
#     ] * 5  # First 5 bars in muted_green, next 5 in coolwarm_red

#     # Create the count plot
#     ax = sns.countplot(data=df_melted, x="Columns", palette=palette)
#     # Set plot background color
#     # fig.patch.set_facecolor(".1")  # Set the background of the figure
#     # ax.set_facecolor(".1")  # Set the background of the axes
#     # Set title and labels with a lighter color for visibility
#     ax.set_title("Number of Comments by Topic")
#     ax.set_xlabel("Topic")
#     ax.set_ylabel("Count of Comments")
#     # Change the color of the ticks and tick labels
#     # ax.tick_params(colors="white")
#     # Change the color of the axes' spines
#     # for spine in ax.spines.values():
#     #     spine.set_edgecolor("white")
#     # Adding the count above each bar
#     for p in ax.patches:
#         ax.annotate(
#             f"{int(p.get_height())}",
#             (p.get_x() + p.get_width() / 2.0, p.get_height()),
#             ha="center",
#             va="center",
#             fontsize=10,
#             # color="white",
#             xytext=(0, 5),
#             textcoords="offset points",
#         )
#     # Calculate total counts for the first 5 and next 5 topics
#     first_5_total = df_melted[
#         df_melted["Columns"].isin(df_melted["Columns"].unique()[:5])
#     ]["Count"].sum()
#     next_5_total = df_melted[
#         df_melted["Columns"].isin(df_melted["Columns"].unique()[5:])
#     ]["Count"].sum()
#     # Calculate percentages
#     positive_perc = first_5_total / (first_5_total + next_5_total) * 100
#     negative_perc = next_5_total / (first_5_total + next_5_total) * 100
#     # Infographic text
#     ax.text(
#         7.8,
#         ax.get_ylim()[1] * 0.92,
#         f"POS : {positive_perc:.4g}%\nNEG : {negative_perc:.4g}%",
#         # color="white",
#         fontweight="bold",
#     )

#     # Display the plot in Streamlit
#     st.pyplot(fig, use_container_width=True)


if __name__ == "__main__":
    main()
