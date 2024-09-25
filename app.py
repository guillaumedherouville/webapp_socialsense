import streamlit as st
import pandas as pd
from functools import partial
import concurrent.futures
import time
import re
from topic_summarization import comments_summarizer, Claude, ChatGPT


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
    all_resp = comments_summarizer(df, movie_info_str, Claude)
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


def display_summary(resp_list, sport):
    content = "trailer/film" if sport == False else "clip/video"
    st.markdown(f"#### Aspects of the {content} that commenters like:")
    likes = "".join([f"{item}\n" for item in resp_list[:5]])
    st.markdown(likes)

    st.markdown(f"#### Aspects of the {content} that commenters dislike:")
    dislikes = "".join([f"{item}\n" for item in resp_list[5:10]])
    st.markdown(dislikes)


def display_selected_topic(summary, comments_topics_df):
    topic = st.selectbox(
        "Select a topic for comments breakdown", summary, label_visibility="collapsed"
    )
    # csv_data = comments_topics_df.to_csv(index=False)
    # st.download_button(
    #     label="Download data as CSV",
    #     data=csv_data,
    #     file_name="comments_data.csv",
    # )
    if topic:
        i = int(summary.index(topic))
        temp = comments_topics_df.set_index("0")
        john = temp[temp.iloc[:, i] != 0]
        john = john.reset_index()
        if len(john) == 0:
            st.write("No comment found matching this topic")
        else:
            with st.container(height=300, border=True):
                comment_block = "\n\n".join(john.iloc[:, 0].values)
                st.markdown(comment_block)


def filter_topics(comments_topics_df):
    temp = comments_topics_df.set_index("0").dropna()
    temp = temp.loc[:, temp.sum(axis=0) > len(temp.loc[temp.sum(axis=1) == 1]) * 0.05]
    index_list = temp.columns.tolist()
    index_list = [int(i) for i in index_list]
    original_positions = [st.session_state.resp_list[i - 1] for i in index_list]
    return original_positions
