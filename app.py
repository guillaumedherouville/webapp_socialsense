import streamlit as st

st.set_page_config(page_title="SocialSense by Jumpcut", layout="wide")
import time
from processing import (
    generate_comments,
    df_character_cleaning,
    get_comments_sentiment,
    comparison_table,
    create_entities_df,
    TOTAL_SUMMARIZER,
    create_movie_info,
    generate_summary_marketing,
    process_comments_in_batches,
)
from config import movies
import re
from visualization import sentiment_viz, emotion_viz, display_comments_by_topic


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


@st.cache_data
def summarize_comments(df, movie_info_str):
    all_resp = TOTAL_SUMMARIZER(df, 3900, movie_info_str)
    resp_list = [item for item in all_resp.splitlines() if item]
    return resp_list


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
    st.markdown("\n".join(resp_list_mark))


def main():
    # Initialize session state variables
    if "video_id" not in st.session_state:
        st.session_state.video_id = None
    if "movie_id" not in st.session_state:
        st.session_state.movie_id = None
    if "comments" not in st.session_state:
        st.session_state.comments = None
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
            st.session_state.comments = generate_comments(
                st.session_state.video_id, st.secrets["YT_KEY"], max_comments=1_000
            )
            log_progress("Cleaning comments...", start_time)
            st.session_state.comments = df_character_cleaning(st.session_state.comments)
            log_progress("Calculating sentiment scores...", start_time)
            st.session_state.all_scores = get_comments_sentiment(
                st.session_state.comments
            )
            log_progress("Creating comparison table...", start_time)
            st.session_state.table = comparison_table(
                st.session_state.all_scores, st.session_state.movie_id, movies
            )
            st.session_state.first_analysis_complete = True

    if st.session_state.get("first_analysis_complete", False):
        st.header("Movie Sentiment and Emotion Analysis 🎈")
        st.subheader("Sentiment Analysis")
        col1, col2 = st.columns(2)
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
        col1.text("")
        col1.dataframe(
            st.session_state.table.set_index("Title")[
                ["sadness", "joy", "love", "anger", "fear", "surprise"]
            ]
        )
        with col2:
            emotion_viz(st.session_state.table)

        st.header("Advanced Topic Analysis 🔎")
        _, st.session_state.entities_df = create_entities_df(st.session_state.movie_id)
        st.session_state.movie_info_str = create_movie_info(
            st.session_state.movie_id, st.session_state.entities_df
        )
        st.session_state.resp_list = summarize_comments(
            st.session_state.comments, st.session_state.movie_info_str
        )
        display_summary(st.session_state.resp_list)
        st.subheader("Marketing Actions Recommendations 🛠️")
        display_marketing(st.session_state.resp_list, st.session_state.movie_info_str)
        comments_topics_df = process_comments_in_batches(
            st.session_state.comments, st.session_state.resp_list, batch_size=50
        )
        st.subheader("Breakdown of Comments by Topic")
        display_comments_by_topic(comments_topics_df)


if __name__ == "__main__":
    main()
