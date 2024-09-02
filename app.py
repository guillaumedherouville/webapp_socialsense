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
from agentic import marketing_process, get_movie_info

goals = {
    "Awareness": "Awareness : we are looking for marketing tactics which will generate awareness for our movie, and make it known to a large audience",
    "Conversion to socials": "Conversion to socials : we are looking for marketing tactics which will generate engagement on social medias",
    "Conversion to viewership": "Conversion to viewership: we are looking for marketing tactics which will directly convert to viewership (i.e. in theatres or streaming services)",
}


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
    if "trailers" not in st.session_state:
        st.session_state.trailers = []
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
    if "table" not in st.session_state:
        st.session_state.table = None
    if "resp_list" not in st.session_state:
        st.session_state.resp_list = None
    if "marketing_actions" not in st.session_state:
        st.session_state.marketing_actions = None
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "context" not in st.session_state:
        st.session_state.context = None
    if "go" not in st.session_state:
        st.session_state.go = None
    if "value" not in st.session_state:
        st.session_state.value = None

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
        number = st.text_input("Number of trailers?", "1")
        if number:
            if number.isdigit():
                number = int(number)
            else:
                st.error("Please enter a valid number")
            for i in range(number):
                youtube_ref = st.text_input(f"Youtube video id or link {i+1}")
                if youtube_ref:
                    video_id = extract_youtube_id(youtube_ref)
                    if video_id is None:
                        st.error("Please enter a valid YouTube ref")
                    else:
                        if video_id not in st.session_state.trailers:
                            st.session_state.trailers.append(video_id)

    with col2:
        imdb_ref = st.text_input("IMDB movie id or link")
        if imdb_ref:
            st.session_state.movie_id = extract_imdb_id(imdb_ref)
            if st.session_state.movie_id is None:
                st.error("Please enter a valid IMDB ref")

    col1, col2, col3, col4, col41, col5, col51 = st.columns(
        [2, 2, 2.1, 1.3, 2, 1.5, 2], vertical_alignment="center"
    )
    st.sidebar.markdown("**Progress**")
    col1.toggle("Sentiment graphs", False, key="sentiment")
    col2.toggle("Match comments", True, key="topic_match")
    col3.toggle("Marketing_standard", False, key="marketing")
    col4.write("Objective :")
    goal = col41.selectbox(
        "Marketing goal",
        ["Awareness", "Conversion to socials", "Conversion to viewership"],
        label_visibility="collapsed",
    )
    st.session_state.goal = goals[goal]
    col5.write("Time horizon :")
    col51.selectbox(
        "Time horizon",
        ["9 months and more", "6 months", "3 months and less"],
        key="time",
        label_visibility="collapsed",
    )
    if st.button("Submit"):
        with st.spinner(
            "Processing... (see progress in sidebar - average time 3-5mins)"
        ):
            st.session_state.start_time = time.time()
        log_progress("Extracting comments...", st.session_state.start_time)
        all_comments = []
        for video_id in st.session_state.trailers:
            comments = generate_comments(
                video_id,
                st.secrets["YT_KEY"],
                max_comments=2_000,  # to cover for non-english
            )
            all_comments.extend(comments)
        st.session_state.trailers = []
        st.session_state.comments = all_comments
        log_progress("Cleaning comments...", st.session_state.start_time)
        st.session_state.comments = df_character_cleaning(st.session_state.comments)
        st.write(f"Total comments: {len(st.session_state.comments)}")

    if st.session_state.comments is not None:
        if st.session_state.sentiment == True:
            log_progress("Calculating sentiment scores...", st.session_state.start_time)
            st.session_state.all_scores = get_comments_sentiment(
                st.session_state.comments
            )
            log_progress("Creating comparison table...", st.session_state.start_time)
            st.session_state.table = comparison_table(
                st.session_state.all_scores, st.session_state.movie_id, movies
            )

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
        perplexity_info = "" #get_movie_info(st.session_state.movie_id)
        _, st.session_state.entities_df = create_entities_df(st.session_state.movie_id)
        st.session_state.movie_info_str = create_movie_info(
            st.session_state.movie_id, st.session_state.entities_df
        )

        st.session_state.context = (
            perplexity_info + "\n\n" + st.session_state.movie_info_str
        )
        st.session_state.value = st.session_state.context
        context = st.text_area(
            "Context found:", value=st.session_state.value, height=400
        )
        if st.button("Confirm"):
            st.session_state.context = context
            st.session_state.go = True
        if st.session_state.go == True:
            log_progress("Generating summary...", st.session_state.start_time)
            st.session_state.resp_list = summarize_comments(
                st.session_state.comments, st.session_state.context
            )
            display_summary(st.session_state.resp_list)
            if st.session_state.topic_match == True:
                log_progress(
                    "Matching comments to topics...", st.session_state.start_time
                )
                comments_topics_df = process_comments_in_batches(
                    st.session_state.comments,
                    st.session_state.resp_list,
                    batch_size=min(len(st.session_state.comments) // 10, 50),
                )
                log_progress("Done!", st.session_state.start_time)
                # st.subheader("Breakdown of comments by topic")
                st.markdown("#### Breakdown of comments by topic:")
                col1, col2 = st.columns(2, vertical_alignment="center")
                # _, col2, _ = st.columns([1, 3, 1])
                with col2:
                    display_comments_by_topic(comments_topics_df)
                with col1:
                    display_selected_topic(
                        st.session_state.resp_list, comments_topics_df
                    )

            log_progress("Suggesting marketing actions...", st.session_state.start_time)
            if st.session_state.marketing == True:
                st.session_state.marketing_actions = generate_summary_marketing(
                    st.session_state.resp_list, st.session_state.context
                )
                st.subheader("Marketing Actions Recommendations 🛠️")
                st.markdown("\n".join(st.session_state.marketing_actions.splitlines()))
            else:
                if st.session_state.topic_match == True:
                    st.write("Round 0:")
                    st.session_state.marketing_actions = generate_summary_marketing(
                        st.session_state.resp_list, st.session_state.context
                    )
                    st.subheader("Marketing Actions Recommendations 🛠️")
                    st.markdown("#### Vanilla suggestions")
                    st.markdown(
                        "\n".join(st.session_state.marketing_actions.splitlines())
                    )
                    marketing_process(
                        filter_topics(comments_topics_df),
                        st.session_state.context,
                        st.session_state.movie_id,
                        st.session_state.goal,
                        st.session_state.time,
                    )
                else:
                    st.write(
                        "Please approve topic match to proceed with marketing actions"
                    )


if __name__ == "__main__":
    main()
