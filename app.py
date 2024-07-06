import streamlit as st
import time
from processing import (
    generate_comments_df,
    df_character_cleaning,
    get_comments_sentiment,
    comparison_table,
    create_entities_df,
    find_mentioned_entities,
    sentiments_df,
    entities_table,
)
from config import movies


def log_progress(container, message, start_time):
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    time_str = f"{minutes:02d}:{seconds:02d}"
    container.write(f"[{time_str}] {message}")


def main():

    st.set_page_config(page_title="Product Demo", layout="wide")

    st.title("Product Demo")

    col1, col2 = st.columns(2)
    with col1:
        video_id = st.text_input("Video ID", key="video_id")

    with col2:
        movie_id = st.text_input("Movie ID", key="movie_id")

    progress_container = col2.container()

    if st.button("Submit"):
        st.write(f"Video ID: {video_id}")
        st.write(f"Movie ID: {movie_id}")

        with st.spinner("Processing..."):
            start_time = time.time()

            log_progress(progress_container, "Extracting comments...", start_time)
            comments = generate_comments_df(video_id)

            log_progress(progress_container, "Cleaning comments...", start_time)
            comments = df_character_cleaning(comments)
            st.write(comments.head(5))

            log_progress(
                progress_container, "Calculating sentiment scores...", start_time
            )
            all_scores = get_comments_sentiment(comments["comment"].tolist())

            log_progress(progress_container, "Creating comparison table...", start_time)
            temp = comparison_table(all_scores, movie_id, movies)
            st.dataframe(temp)

            log_progress(progress_container, "Extracting entities...", start_time)
            to_summarize = comments["comment"].astype(str).tolist()
            entities = create_entities_df(movie_id)

            log_progress(progress_container, "Finding entities...", start_time)
            entity_mentions = find_mentioned_entities(to_summarize, entities)

            log_progress(
                progress_container, "Matching sentiments to entities...", start_time
            )
            sentiment_df = sentiments_df(to_summarize)
            temp = entities_table(entity_mentions, sentiment_df)
            st.dataframe(temp)
            log_progress(progress_container, "Done!", start_time)


if __name__ == "__main__":
    main()
