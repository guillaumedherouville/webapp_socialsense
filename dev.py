import streamlit as st
import pandas as pd
import random
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
from visualization import sentiment_viz, emotion_viz, display_comments_by_topic
from sport import (
    sports_table,
    summarize_sports,
    sports_marketing_process,
    topic_attribution_sports,
    sports_goals,
)
from agentic import marketing_process, goals
from app import (
    extract_youtube_id,
    extract_imdb_id,
    log_progress,
    summarize_comments,
    process_comments_in_batches,
    display_summary,
    display_selected_topic,
    filter_topics,
)


def dev_page():
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
            tiktok = st.file_uploader("Upload CSV comments", type=["csv"])
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
            if st.session_state.sport:
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
                    st.session_state.comments += st.session_state.tiktok
            st.session_state.first_analysis = True
            st.session_state.value = 1
            st.session_state.devalue = 1

    if (
        st.session_state.value == 1
        and st.session_state.devalue == 1
        and st.session_state.first_analysis is True
    ):
        log_progress("Cleaning comments...", st.session_state.start_time)
        st.session_state.comments = df_character_cleaning(
            st.session_state.comments[
                :1000
            ]  ## NOT NEEDED WHEN WE DON'T FILTER FOR ENGLISH ONLY
        )

        st.write("Number of comments processed:", len(st.session_state.comments))
        if st.session_state.tiktok is not None:
            st.write("Preview of tiktok comments:")
            st.table(st.session_state.tiktok[:10])
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
            st.markdown("Context")
            st.write(st.session_state.movie_info_str)
            st.session_state.resp_list = summarize_comments(
                st.session_state.comments, st.session_state.movie_info_str
            )
        display_summary(st.session_state.resp_list, st.session_state.sport)
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
        st.markdown("### No marketing : currently working on topic summarization only")
        # log_progress("Suggesting marketing actions...", st.session_state.start_time)
        # if st.session_state.sport:
        #     st.session_state.marketing_actions = sports_marketing_process(
        #         filter_topics(comments_topics_df),
        #         st.session_state.goal,
        #     )
        # else:
        #     marketing_process(
        #         filter_topics(comments_topics_df),
        #         st.session_state.movie_info_str,
        #         st.session_state.goal,
        #     )
