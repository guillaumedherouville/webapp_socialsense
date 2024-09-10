import streamlit as st
import pandas as pd
import os
from functools import partial
import concurrent.futures
import random


os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(page_title="SocialSense by Jumpcut", layout="wide")
import time
from processing import (
    generate_comments,
    df_character_cleaning,
    get_comments_sentiment,
    comparison_table,
    create_entities_df,
    create_movie_info,
    match_topics_comments,
)
from config import movies, wwe
import re
from visualization import sentiment_viz, emotion_viz, display_comments_by_topic
from sport import (
    sports_table,
    summarize_sports,
    sports_marketing_process,
    topic_attribution_sports,
    sports_goals,
)
from agentic import comments_summarizer, marketing_process, goals


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


@st.cache_data(show_spinner=False)
def summarize_comments(df, movie_info_str):
    all_resp = comments_summarizer(df, movie_info_str)
    resp_list = [item for item in all_resp.splitlines() if item]
    return resp_list


@st.cache_data(show_spinner=False)
def process_comments_in_batches(comments, summary, _match_fctn, batch_size=50):
    batches = []
    for i in range(0, len(comments), batch_size):
        batches.append(comments[i : i + batch_size])
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(partial(_match_fctn, all_resp=summary), batches))
    flattened_result = [
        item for sublist in results if sublist is not None for item in sublist
    ]
    df = pd.DataFrame(flattened_result)
    return df


def display_summary(resp_list):
    st.markdown("#### Aspects of the trailer/film that commenters like:")
    likes = "".join([f"{item}\n" for item in resp_list[:5]])
    st.markdown(likes)

    st.markdown("#### Aspects of the trailer/film that commenters dislike:")
    dislikes = "".join([f"{item}\n" for item in resp_list[5:10]])
    st.markdown(dislikes)


def display_selected_topic(summary, comments_topics_df):
    topic = st.selectbox(
        "Select a topic for comments breakdown", summary, label_visibility="collapsed"
    )
    if topic:
        i = int(summary.index(topic))
        temp = comments_topics_df.set_index("0")
        john = temp[temp.iloc[:, i] == 1]
        john = john.reset_index()
        if len(john) == 0:
            st.write("No comment found matching this topic")
        else:
            with st.container(height=300, border=True):
                for idx, row in john.iterrows():
                    st.write(row[0])


def filter_topics(comments_topics_df):
    temp = comments_topics_df.set_index("0").dropna()
    temp = temp.loc[:, temp.sum(axis=0) > len(temp.loc[temp.sum(axis=1) == 1]) * 0.05]
    index_list = temp.columns.tolist()
    index_list = [int(i) for i in index_list]
    original_positions = [st.session_state.resp_list[i - 1] for i in index_list]
    return original_positions


def main():
    # Initialize session state variables
    if "video_id" not in st.session_state:
        st.session_state.video_id = None
    if "movie_id" not in st.session_state:
        st.session_state.movie_id = None
    if "comments" not in st.session_state:
        st.session_state.comments = []
    if "all_scores" not in st.session_state:
        st.session_state.all_scores = None
    if "entities_df" not in st.session_state:
        st.session_state.entities_df = None
    if "sorted_values" not in st.session_state:
        st.session_state.sorted_values = None
    if "sorted_values_2" not in st.session_state:
        st.session_state.sorted_values_2 = None
    if "temp" not in st.session_state:
        st.session_state.temp = None
    if "movie_info_str" not in st.session_state:
        st.session_state.movie_info_str = None
    if "first_analysis_complete" not in st.session_state:
        st.session_state.first_analysis_complete = False
    if "table" not in st.session_state:
        st.session_state.table = None
    if "resp_list" not in st.session_state:
        st.session_state.resp_list = None
    if "marketing_actions" not in st.session_state:
        st.session_state.marketing_actions = None
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "sport" not in st.session_state:
        st.session_state.sport = False
    if "tiktok" not in st.session_state:
        st.session_state.tiktok = None
    if "goal" not in st.session_state:
        st.session_state.goal = None
    if "trailers" not in st.session_state:
        st.session_state.trailers = []

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
        '<div class="center-text"><h1>{}</h1></div>'.format(
            "SocialSense by Jumpcut 🎬"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="center-text"><h2>{}</h2></div>'.format(
            "Social Intelligence and Marketing Optimization"
        ),
        unsafe_allow_html=True,
    )

    st.sidebar.toggle("Sport", False, key="sport")
    col1, col2, col3 = st.columns([1, 1, 2], vertical_alignment="center")
    if st.session_state.sport == False:
        with col1:
            youtube_ref = st.text_input("Youtube video id or link")
            if youtube_ref:
                st.session_state.video_id = extract_youtube_id(youtube_ref)
                if st.session_state.video_id is None:
                    st.error("Please enter a valid YouTube ref")
    else:
        with col1:
            st.session_state.number = st.text_input("Number of trailers?", "1")
            if st.session_state.number:
                if st.session_state.number.isdigit():
                    st.session_state.number = int(st.session_state.number)
                else:
                    st.error("Please enter a valid number")
                for i in range(st.session_state.number):
                    youtube_ref = st.text_input(f"Youtube video id or link {i+1}")
                    if youtube_ref:
                        video_id = extract_youtube_id(youtube_ref)
                        if video_id is None:
                            st.error("Please enter a valid YouTube ref")
                        else:
                            if video_id not in st.session_state.trailers:
                                st.session_state.trailers.append(video_id)

    if st.session_state.sport == False:
        with col2:
            imdb_ref = st.text_input("IMDB movie id or link")
            if imdb_ref:
                st.session_state.movie_id = extract_imdb_id(imdb_ref)
                if st.session_state.movie_id is None:
                    st.error("Please enter a valid IMDB ref")

    if st.session_state.sport == False:
        with col3:
            tiktok = st.file_uploader("Upload TikTok comments", type=["csv"])
            if tiktok:
                tiktok = pd.read_csv(tiktok)
                st.session_state.tiktok = tiktok["Comment"].to_list()
                if st.session_state.tiktok is None:
                    st.error("Please provide a 'Comment' column in your csv file")

    col1, col2, col3 = st.columns(3, vertical_alignment="center")
    col2.write("Objective :")
    goal = col3.selectbox(
        "Marketing goal",
        (
            ["Awareness", "Conversion to socials", "Conversion to viewership"]
            if not st.session_state.sport
            else ["Youtube videos", "Character development", "General marketing"]
        ),
        label_visibility="collapsed",
    )
    st.session_state.goal = (
        goals[goal] if not st.session_state.sport else sports_goals[goal]
    )
    if col1.button("Submit"):
        st.sidebar.markdown("**Progress**")
        with st.spinner(
            "Processing... (see progress in sidebar - average time 3-5mins)"
        ):
            st.session_state.start_time = time.time()
            log_progress("Extracting comments...", st.session_state.start_time)
            if st.session_state.sport and st.session_state.number > 1:
                all_comments = []
                for video_id in st.session_state.trailers:
                    comments = generate_comments(
                        video_id,
                        st.secrets["YT_KEY"],
                        max_comments=1_000,
                    )
                    all_comments.extend(comments)
                random.shuffle(all_comments)
                st.session_state.comments = all_comments[
                    :1000
                ]  ## PUT SOME PADDING IF WE DO FILTER FOR ENGLISH COMMENTS
            else:
                if st.session_state.video_id:
                    st.session_state.comments = generate_comments(
                        st.session_state.video_id,
                        st.secrets["YT_KEY"],
                        max_comments=1_000,
                    )
            if st.session_state.tiktok is not None:
                st.session_state.comments = (
                    st.session_state.comments + st.session_state.tiktok
                )
            log_progress("Cleaning comments...", st.session_state.start_time)
            st.session_state.comments = df_character_cleaning(
                st.session_state.comments[
                    :1000
                ]  ## NOT NEEDED WHEN WE DON'T FILTER FOR ENGLISH ONLY
            )
            st.write("Number of comments processed:", len(st.session_state.comments))
            log_progress("Calculating sentiment scores...", st.session_state.start_time)
            st.session_state.all_scores = get_comments_sentiment(
                st.session_state.comments
            )
            log_progress("Creating comparison table...", st.session_state.start_time)
            if st.session_state.sport:
                st.session_state.table = sports_table(st.session_state.all_scores, wwe)
            else:
                st.session_state.table = comparison_table(
                    st.session_state.all_scores, st.session_state.movie_id, movies
                )
            st.session_state.first_analysis_complete = True

    if st.session_state.get("first_analysis_complete", False):
        st.write("Number of comments processed:", len(st.session_state.comments))
        if st.session_state.tiktok:
            st.write("Preview of tiktok comments:")
            st.table(st.session_state.tiktok[:10])
        st.header("Movie Sentiment and Emotion Analysis 🎈")
        st.subheader("Sentiment Analysis")
        col1, col2 = st.columns(2, vertical_alignment="center")
        col1.dataframe(
            st.session_state.table.set_index("Title")[
                ["negative", "neutral", "positive"]
            ]
        )
        with col2:
            sentiment_viz(st.session_state.table)
        st.subheader("Emotion Analysis")
        col1, col2 = st.columns(2, vertical_alignment="center")
        col1.dataframe(
            st.session_state.table.set_index("Title")[
                ["sadness", "joy", "love", "anger", "fear", "surprise"]
            ]
        )
        with col2:
            emotion_viz(st.session_state.table)

        st.header("Advanced Topic Analysis 🔎")
        log_progress("Generating summary...", st.session_state.start_time)
        if st.session_state.sport:
            if st.session_state.number == 1:
                st.session_state.resp_list = summarize_sports(st.session_state.comments)
            else:
                st.session_state.resp_list = summarize_sports(
                    st.session_state.comments, True
                )
        else:
            _, st.session_state.entities_df = create_entities_df(
                st.session_state.movie_id
            )
            st.session_state.movie_info_str = create_movie_info(
                st.session_state.movie_id, st.session_state.entities_df
            )
            st.session_state.resp_list = summarize_comments(
                st.session_state.comments, st.session_state.movie_info_str
            )
        display_summary(st.session_state.resp_list)
        log_progress("Matching comments to topics...", st.session_state.start_time)
        comments_topics_df = process_comments_in_batches(
            st.session_state.comments,
            st.session_state.resp_list,
            (
                match_topics_comments
                if not st.session_state.sport
                else topic_attribution_sports
            ),
            batch_size=min(len(st.session_state.comments) // 10, 50),
        )
        st.subheader("Breakdown of comments by topic 📍")
        col1, col2 = st.columns(2, vertical_alignment="center")
        with col2:
            display_comments_by_topic(comments_topics_df)
        with col1:
            display_selected_topic(st.session_state.resp_list, comments_topics_df)
        log_progress("Suggesting marketing actions...", st.session_state.start_time)
        if st.session_state.sport:
            st.session_state.marketing_actions = sports_marketing_process(
                filter_topics(comments_topics_df),
                st.session_state.goal,
            )
        else:
            marketing_process(
                filter_topics(comments_topics_df),
                st.session_state.movie_info_str,
                st.session_state.goal,
            )


if __name__ == "__main__":
    main()
