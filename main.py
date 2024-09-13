import streamlit as st
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(page_title="SocialSense by Jumpcut", layout="wide")
from dev import dev_page
from prod import prod_page


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
    if "value" not in st.session_state:
        st.session_state.value = -1
    if "devalue" not in st.session_state:
        st.session_state.devalue = -1
    if "first_analysis" not in st.session_state:
        st.session_state.first_analysis = False
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
    emoji = "🎬" if st.session_state.sport is False else "🤼🏽"
    st.markdown(
        '<div class="center-text"><h1>{}</h1></div>'.format(
            f"SocialSense by Jumpcut {emoji}"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="center-text"><h2>{}</h2></div>'.format(
            "Social Intelligence and Marketing Optimization"
        ),
        unsafe_allow_html=True,
    )

    def sport_change():
        st.session_state.value *= -1

    st.sidebar.toggle("Sport", False, key="sport", on_change=sport_change)
    st.sidebar.toggle("DEV", False, key="dev")
    if st.session_state.dev == False:
        prod_page()
    else:
        dev_page()


if __name__ == "__main__":
    main()
