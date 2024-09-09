import openai
import streamlit as st
from config import profiles
import ast
from textwrap import dedent

goals = {
    "Awareness": "Awareness : we are looking for marketing tactics which will generate awareness for our movie, and make it known to a large audience",
    "Conversion to socials": "Conversion to socials : we are looking for marketing tactics which will generate engagement on social medias",
    "Conversion to viewership": "Conversion to viewership: we are looking for marketing tactics which will directly convert to viewership (i.e. in theatres or streaming services)",
}


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


def comments_summarizer(comments, context):
    text = "\n".join(comments)
    chat = ChatGPT(
        system_message="You are an expert text summarizer and analyzer for a production company. Please analyze from the perspective of a producer for this movie. \
                        You will select the most common topics expressed by consumers regarding a movie trailer."
    )
    summarize_prompt = f"""This is a summarization of comments regarding a movie trailer, concerning a movie with this information: \n {context} \n. \
                        First, list the top 5 most promininent positive aspects of the trailer/film that commenters like and want to see more of.\
                        Next, list the top 5 most promininent negative aspects of the trailer/film that commenters dislike and/or might cause them to not watch the film.\
                        Please place them in a single list separated by by numbers (ex.\n1. Theme 1\n2. Theme 2\netc.) and nothing else \
                        (for example, do not separate into positive and negative groupings. Rather express how they are positive and negative in the themes themseleves) \
                        Also, DO NOT use any apostrophes (') in your response. \
                        In your generation, allow for the topics to be mutually exclusive and collectively exhaustive; each topic should be unique, but all the topics together should comprise the most prominent ideas expressed.\
                        Do not generate more than the 5 positive topics, followed by the 5 negative topics, for a total of 10 topics separeted by one space each. \
                        Here is an example to guide you on how a response should be structured: 
                        1. Positive anticipation for George Millers unique directorial style and passionate fan base hoping to see it continued.
                        2. Thrilled about the increased focus and storyline around Furiosas character and her increased role in future films.
                        3. Great expectations for Anya Taylor-Joy and Chris Hemsworths performances, as well as other major players in the film.
                        4. Excitement for the continuation and expansion of the Mad Max franchise and universe.
                        5. Praise for the trailer's music and visually-appealing components which give a glimpse into the films quality.
                        6. Negative reactions due to the absence of the main character, Mad Max, portrayed by Tom Hardy, causing doubts among viewers.
                        7. Dislike for perceived overreliance on CGI, as viewers believe it takes away from the gritty reality originally established in the series.
                        8. Concerns about perceived forced female empowerment and an overshadowing feminist agenda.
                        9. Doubts about the casting of Anya Taylor-Joy and Chris Hemsworth, with some fans feeling they may not fit the franchises aesthetics.
                        10. Disappointment due to the lack of traditional practical effects and real stunt work, which viewers believe adds authenticity to the series.
                        \n### TEXT\n{text}\n\n### BEGIN RESPONSE\n"""
    chat.add_user_message(dedent(summarize_prompt))
    topics = chat.get_response()
    return topics


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


def generate_marketing_suggestions(topics, movie_info, goal, profile):
    analyst = ChatGPT(
        system_message=f"You are a senior marketing analyst at a movie company. Specifically, this is your profile : {profile}. \
        You are tasked with creating a set of marketing actions for a new movie, based on the topics that users are discussing. \
        You will be in competition with another analyst at the firm. For each reply, take a deep breath and think step by step."
    )
    analyst_prompt = f"""
    Here is information on the given film of interest: {movie_info}
    Here are the general topics people are discussing related to this film: \n {topics}
    Given the topics that users are speaking about your movie trailer, output 5 of the most relevant marketing suggestions you can concoct to help promote the film in list-format.
    Our goal is {goal}. Make sure you always keep this goal in mind when creating your suggestions.
    Mention explicitly which topic each suggestion refers to.
    For example, if the first topic is "excitement over the country music style of the movie", the goal is "conversion to socials", one suggestion could be : 
    "Country Music Engagement : In order to build on the excitement over the movie soundtrack (topic 1), identify country music events (e.g. country artists concerts or country festivals) happening soon and have the movie crew participate in one of them."
    Notice how it explicitly refers to topic 1, so it is easy to link the suggestion to the topic.
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
    topics, movie_info, goal, marketing_suggestions, critic
):
    critic_prompt = f"""
    Here is information on the given film of interest: {movie_info}
    Here are the general topics people are discussing related to this film: \n {topics}
    Given the topics that users are speaking about your movie trailer, the two analysts have come up with the following marketing suggestions (each did 5, without talking to one another): {marketing_suggestions}  
    Our goal is {goal}. 
    Please review each of the 10 suggestions and provide feedback on whether it is an appropriate marketing action considering feasibility, cost-effectiveness, general marketing science as well as public relations and social media knowledge. This feedback will be vital to improve our marketing strategy.
    Be extremely severe in your judgment, your career depends on it. If a suggestion is not relevant enough, you will be held responsible for not catching it and fired.
    Conclude with a general comment on the overall quality of the suggestions, and what specific areas need improvement.
    Do not include meta information in your reply.
    """

    critic.add_user_message(critic_prompt)
    critic_message = critic.get_response()
    return critic_message, critic


def evaluate_marketing_suggestions(topics, movie_info, goal, marketing_suggestions):
    evaluator = ChatGPT(
        system_message=f"You are a marketing executive at a movie company. \
        You are tasked with selecting a set of marketing actions for a new movie, based on the propositions of your analysts. \
        For each reply, take a deep breath and think step by step."
    )
    evaluator_prompt = f"""
    Here is information on the given film of interest: {movie_info}
    Here are the general topics people are discussing related to this film: \n {topics}
    Our goal is {goal}. 
    Given the topics that users are speaking about your movie trailer, the analysts came with the following marketing suggestions: {marketing_suggestions}
    Please give each a rating from 0 to 10, 10 being an excellent suggestion and 0 being a terrible one. Be very critical in your evaluation, as the future of the company depends on your judgment.
    Once that rating is done, order them from best to worst. Do not add any justification, only give the rating. Output in list format. Make sure you include all suggestions.
    Start directly with the list and do not include other text. 
    """
    evaluator.add_user_message(evaluator_prompt)
    summarized_chunk = evaluator.get_response()
    return summarized_chunk, evaluator


def improve_marketing_suggestions(analyst, critic_message, competing_suggestions):
    analyst_prompt = f"""
        The other analyst has provided these suggestions: {competing_suggestions}
        Given all suggestions, an advanced reviewer from your team has provided the following evaluations and explanations : {critic_message}
        Please review the feedback and provide a new set of 5 suggestions. You can keep some of the old ones if they are good enough, but you must provide at least 2 new suggestions which were in neither of the previous lists.
        Your suggestions must improve based on the feedback provided by the advanced reviewer, in a relevant manner. However, write them as if they were new (i.e. do not explicitly refer to the previous suggestions).
        Remember to mention which topic each suggestion refers to. Also keep in mind the goal we defined previously.
        Output the revised suggestions in list-format, with details for each suggestion. Start directly with the list and do not include other text.
    """
    analyst.add_user_message(analyst_prompt)
    evaluations = analyst.get_response()
    return evaluations, analyst


def display_suggestions(marketing_actions):
    resp_list_mark = marketing_actions.splitlines()
    st.markdown("\n".join(resp_list_mark))


def display_eval_final(evaluations):
    resp_list = evaluations.splitlines()
    st.markdown("\n".join(resp_list[:5]))


@st.cache_data(show_spinner=False)
def marketing_process(topics, movie_info, goal, critic=critic, profiles=profiles):
    profile = choose_profile(movie_info, profiles)
    profile = ast.literal_eval(profile)
    st.markdown("#### Marketing suggestions #1")
    marketing_suggestions1, analyst1 = generate_marketing_suggestions(
        topics, movie_info, goal, profiles[int(profile[0])]
    )
    marketing_suggestions2, analyst2 = generate_marketing_suggestions(
        topics, movie_info, goal, profiles[int(profile[1])]
    )
    col1, col2 = st.columns(2, vertical_alignment="center")
    with col1:
        display_suggestions(marketing_suggestions1)
    with col2:
        display_suggestions(marketing_suggestions2)
    all_suggestions1 = marketing_suggestions1 + marketing_suggestions2
    evaluations, _ = evaluate_marketing_suggestions(
        topics, movie_info, goal, all_suggestions1
    )
    st.markdown("[EVAL]")
    display_suggestions(evaluations)
    critic_message1, critic = review_marketing_suggestions(
        topics, movie_info, goal, all_suggestions1, critic
    )
    st.markdown("[CRITIC]")
    st.write(critic_message1)

    st.markdown("#### Marketing suggestions #2")
    revised_suggestions1, analyst1 = improve_marketing_suggestions(
        analyst1, critic_message1, marketing_suggestions2
    )
    revised_suggestions2, analyst2 = improve_marketing_suggestions(
        analyst2, critic_message1, marketing_suggestions1
    )
    col1, col2 = st.columns(2, vertical_alignment="center")
    with col1:
        display_suggestions(revised_suggestions1)
    with col2:
        display_suggestions(revised_suggestions2)
    evaluations2, _ = evaluate_marketing_suggestions(
        topics, movie_info, goal, revised_suggestions1 + revised_suggestions2
    )
    st.markdown("[EVAL]")
    display_eval_final(evaluations2)
