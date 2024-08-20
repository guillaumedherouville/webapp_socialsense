import openai
import streamlit as st
from imdb import IMDb
from config import profiles
import ast


class ChatGPT:
    def __init__(self, model="gpt-4o-mini", system_message=None):
        self.model = model
        self.client = openai.OpenAI()
        self.default_system_message = {"role": "system", "content": system_message}
        self.messages = [self.default_system_message]

    def add_user_message(self, message, reset_chat=False):
        if reset_chat:
            self.messages = [self.default_system_message]
        self.messages.append({"role": "user", "content": message})

    def get_response(self):
        completion = self.client.chat.completions.create(
            model=self.model, messages=self.messages
        )
        assistant_message = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message


goals = {
    "Awareness": "Awareness : we are looking for marketing tactics which will generate awareness for our movie, and make it known to a large audience",
    "Conversion to socials": "Conversion to socials : we are looking for marketing tactics which will generate engagement on social medias",
    "Conversion to viewership": "Conversion to viewership: we are looking for marketing tactics which will directly convert to viewership (i.e. in theatres or streaming services)",
}


def choose_profile(movie_info, profiles):
    analyst = ChatGPT(
        system_message="You are a staffer at a movie marketing company. You are tasked with assigning certain profiles to certain jobs. For each reply, take a deep breath and think step by step."
    )
    analyst_prompt = f"""
    A new client is coming to you with a movie to market.
    Here is information on the given film: {movie_info}
    Your role is to choose the two most appropriate marketers for this movie, from the following list: {profiles}.
    Your output should be 2 digits in list-format, indicating the position in the list of the relevant profiles. 
    Start directly with the list and include nothing else.
    """
    analyst.add_user_message(analyst_prompt)
    summarized_chunk = analyst.get_response()
    return summarized_chunk


def generate_marketing_suggestions(topics, movie_info, goal, time_horizon, profile):
    analyst = ChatGPT(
        system_message=f"You are a senior marketing analyst at a movie company. Specifically, this is your profile : {profile}. \
        You are tasked with creating a set of marketing actions for a new movie, based on the topics that users are discussing. \
        For each reply, take a deep breath and think step by step."
    )
    analyst_prompt = f"""
    Here is information on the given film of interest: {movie_info}
    Here are the general topics people are discussing related to this film: \n {topics}
    Given the topics that users are speaking about your movie trailer, output 5 of the most relevant marketing suggestions you can concoct to help promote the film in list-format.
    Our goal is {goal}. And the time horizon is {time_horizon}, so please keep this is mind for your suggestions when evaluating their feasibility.
    Mention explicitly which topic each suggestion refers to.
    For example, if the first topic is "excitement over the country music style of the movie", the goal is "conversion to socials" and time horizon "6 months" one suggestion could be : 
    "Country Music Engagement : In order to build on the excitement over the movie soundtrack (topic 1), identify country music events (e.g. country artists concerts or country festivals) happening soon and have the movie crew participate in one of them."
    Notice how it explicetly refers to topic 1, so it is easy to link the suggestion to the topic.
    Please ensure each suggestion is unique; do not repeat similar suggestions times.
    Start directly with the list. 
    """
    analyst.add_user_message(analyst_prompt)
    summarized_chunk = analyst.get_response()
    return summarized_chunk, analyst


critic = ChatGPT(
    system_message="You are a senior marketing associate at a big movie production company. You are tasked with guiding two junior analysts to provide optimal marketing actions for a new movie, based on the topics that users are discussing. \
    For each reply, take a deep breath and think step by step. I will tip you $100."
)


def review_marketing_suggestions(
    topics, movie_info, marketing_suggestions, critic, first_pass=False, final=False
):
    suffix = f"""
    Please review each suggestion and provide feedback on whether it is an appropriate marketing action considering feasibility, cost-effectiveness, general marketing science as well as public relations and social media knowledge. This feedback will be vital to improve our marketing strategy.
    Be extremely severe in your judgment, your career depends on it. If a suggestion is not relevant enough, you will be held responsible for not catching it and fired.
    At the beginning of your review, make sure to provide a clear ranking of the 10 suggestions, from best to worst.
    Conclude with a general comment on the overall quality of the suggestions, and what specific areas need improvement.
    Do not include meta information in your reply.
    """

    if first_pass == True:
        critic_prompt = f"""
        Here is information on the given film of interest: {movie_info}
        Here are the general topics people are discussing related to this film: \n {topics}
        Given the topics that users are speaking about your movie trailer, the two analysts have come up with the following marketing suggestions, in no particular order: {marketing_suggestions}
        {suffix}
        """
    else:
        critic_prompt = f"""
        Given your feedback, the two analysts have revised the marketing suggestions for the film. Here are the updated suggestions: {marketing_suggestions}
        {suffix}"""
    if final == True:
        critic_prompt = f"""
        Given your feedback, the two analysts have revised the marketing suggestions for the film. Here are the final updated suggestions: {marketing_suggestions}
        Please review each suggestion assign it a score based of whether it is an appropriate marketing action considering feasibility, cost-effectiveness, general marketing science as well as public relations and social media knowledge. This feedback will be vital to improve our marketing strategy.
        Be extremely severe in your judgment, your career depends on it. 
        Return the top 5 best unique suggestions, in list format, with the exact same format as it was provided by the analyst.
        Start directly with the list and do not include other text."""

    critic.add_user_message(critic_prompt)
    critic_message = critic.get_response()
    return critic_message, critic


def improve_marketing_suggestions(analyst, critic_message, competing_suggestions):
    analyst_prompt = f"""
        The other analyst has provided these suggestions: {competing_suggestions}
        Given all suggestions, an advanced reviewer from your team has provided the following ranking and explanation : {critic_message}
        Please review the feedback and provide a new set of 5 suggestions. You can keep some of the old ones if they are good enough, but you must provide at least 2 new suggestions which were in neither of the previous lists.
        Your suggestions must improve based on the feedback provided by the advanced reviewer, in a relevant manner.
        Output the revised suggestions in list-format, with details for each suggestion. Start directly with the list and do not include other text.
    """
    analyst.add_user_message(analyst_prompt)
    evaluations = analyst.get_response()
    return evaluations, analyst


def display_suggestions(marketing_actions):
    resp_list_mark = marketing_actions.splitlines()
    st.markdown("\n".join(resp_list_mark))


def get_movie_info(movie_id):
    ia = IMDb()
    movie = ia.get_movie(movie_id)
    messages = [
        {
            "role": "system",
            "content": "You are a marketing research assistant that provides accurate info about upcoming movies.",
        },
        {
            "role": "user",
            "content": f"What 1-2 sentences of context should I know about the (1) plot, (2) cast, (3) relevant cultural info and (4) general categories (do not show but i.e. inde or not, genre, target audience...) related to the upcoming movie  '{movie}'?",
        },
    ]
    client = openai.OpenAI(
        api_key=st.secrets["PERPLEXITY"], base_url="https://api.perplexity.ai"
    )  # chat completion without streaming
    response = client.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online",
        messages=messages,
    )
    return response.choices[0].message.content


@st.cache_data(show_spinner=False)
def marketing_process(
    topics, movie_info, movie_id, goal, time_horizon, critic=critic, profiles=profiles
):
    st.subheader("Marketing Actions Recommendations 🛠️")
    # st.markdown("#### Perplexity info:")
    # info = get_movie_info(movie_id)
    # st.write(info)
    profile = choose_profile(movie_info, profiles)
    st.write(profile)
    profile = ast.literal_eval(profile)
    st.markdown("#### Marketing suggestions #1")
    marketing_suggestions1, analyst1 = generate_marketing_suggestions(
        topics, movie_info, goal, time_horizon, profiles[int(profile[0])]
    )
    marketing_suggestions2, analyst2 = generate_marketing_suggestions(
        topics, movie_info, goal, time_horizon, profiles[int(profile[1])]
    )
    col1, col2 = st.columns(2, vertical_alignment="center")
    with col1:
        display_suggestions(marketing_suggestions1)
    with col2:
        display_suggestions(marketing_suggestions2)
    # all_suggestions1 = marketing_suggestions1 + marketing_suggestions2
    # critic_message1, critic = review_marketing_suggestions(
    #     topics, movie_info, all_suggestions1, critic, first_pass=True
    # )
    # st.write(critic_message1)

    # st.markdown("#### Marketing suggestions #2")
    # revised_suggestions1, analyst1 = improve_marketing_suggestions(
    #     analyst1, critic_message1, marketing_suggestions2
    # )
    # revised_suggestions2, analyst2 = improve_marketing_suggestions(
    #     analyst2, critic_message1, marketing_suggestions1
    # )
    # col1, col2 = st.columns(2, vertical_alignment="center")
    # with col1:
    #     display_suggestions(revised_suggestions1)
    # with col2:
    #     display_suggestions(revised_suggestions2)
    # all_suggestions2 = revised_suggestions1 + revised_suggestions2
    # critic_message2, critic = review_marketing_suggestions(
    #     None,
    #     None,
    #     all_suggestions2,
    #     critic,
    #     None,
    #     final=True,  # delete the last two for previous
    # )
    # st.write(critic_message2)
