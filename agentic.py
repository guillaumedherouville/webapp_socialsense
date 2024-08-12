import openai
import streamlit as st


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


def generate_marketing_suggestions(topics, movie_info):
    analyst_prompt = f"""
    Here is information on the given film of interest: {movie_info}
    Here are the general topics people are discussing related to this film: \n {topics}
    Given the topics that users are speaking about your movie trailer, output 5 of the most relevant marketing suggestions you can concoct to help promote the film in list-format, with details for being included in application to this specific film.
    Please lend creative and specific suggestions to help market this film.
    Please ensure each suggestion is unique; do not repeat similar suggestions multiple times.
    Start directly with the list, do not include other text, and be concise (yet detailed) in your suggestions.
    """
    analyst = ChatGPT(
        system_message="You are a senior marketing analyst at a big movie production company. You are tasked with creating a set of marketing actions for a new movie, based on the topics that users are discussing"
    )
    analyst.add_user_message(analyst_prompt)
    summarized_chunk = analyst.get_response()
    return summarized_chunk, analyst


def evaluate_marketing_suggestions(topics, movie_info, marketing_suggestions):
    evaluation_prompt = f"""
    Here is information on the movie of interest: {movie_info}
    Here are the general topics people are discussing related to this film: \n {topics}
    Given the topics that users are speaking about your movie trailer, your team has come up with the following marketing suggestions : {marketing_suggestions}
    Rate the suggestions from 1-5, with 1 being the least helpful and 5 being the most helpful. Output the ratings in list-format.
    Output should only be 5 numbers, each separated by a \\n and nothing else.
    """
    evaluator = ChatGPT(
        system_message="You are a senior marketing executive at a big movie production company. You are tasked with evaluating a set of marketing actions for a new movie based on the topics that users are discussing."
    )
    evaluator.add_user_message(evaluation_prompt)
    evaluations = evaluator.get_response()
    return evaluations


def review_marketing_suggestions(
    topics, movie_info, marketing_suggestions, critic, first_pass
):
    if first_pass:
        critic_prompt = f"""
        Here is information on the given film of interest: {movie_info}
        Here are the general topics people are discussing related to this film: \n {topics}
        Given the topics that users are speaking about your movie trailer, your analyst has come up with the following marketing suggestions : {marketing_suggestions}
        Please review the suggestions and provide feedback on how they can be improved or expanded upon. Output your feedback in list-format, with explanation for each suggestion.
    """
    else:
        critic_prompt = f"""
        Given your feedback, your analyst has revised the marketing suggestions for the film. Here are the updated suggestions: {marketing_suggestions}
        Please review these new suggestions and provide feedback on how they can be improved or expanded upon. Output your feedback in list-format, with explanation for each suggestion.
    """
    critic.add_user_message(critic_prompt)
    critic_message = critic.get_response()
    return critic_message, critic


def improve_marketing_suggestions(analyst, critic_message):
    analyst_prompt = f"""
        Given your suggestions, an advanced reviewer from your team has provided the following feedback : {critic_message}
        Please review the feedback and make any necessary changes to the marketing suggestions. Output the revised suggestions in list-format, with details for each suggestion.
        Start directly with the list and do not include other text.
    """
    analyst.add_user_message(analyst_prompt)
    evaluations = analyst.get_response()
    return evaluations, analyst


def display_suggestions(marketing_actions):
    resp_list_mark = marketing_actions.splitlines()
    st.markdown("\n".join(resp_list_mark))


def display_evaluations(evaluations):
    st.write("Evaluations:")
    evaluations = " - ".join(
        [f"Suggestion #{i+1}: {evaluations[i]}" for i in range(len(evaluations))]
    )
    st.write(evaluations)


def marketing_process(topics, movie_info):
    st.subheader("Marketing suggestions #1")
    marketing_suggestions, analyst = generate_marketing_suggestions(topics, movie_info)
    display_suggestions(marketing_suggestions)

    evaluations = evaluate_marketing_suggestions(
        topics, movie_info, marketing_suggestions
    )
    display_evaluations(evaluations)

    critic = ChatGPT(
        system_message="You are a senior marketing associate at a big movie production company. You are tasked with guiding a younger a younger analyst to provide optimal marketing actions for a new movie, based on the topics that users are discussing."
    )
    critic_message, critic = review_marketing_suggestions(
        topics, movie_info, marketing_suggestions, critic, first_pass=True
    )

    for i in range(2, 6):
        st.subheader(f"Marketing suggestions #{i}")
        revised_suggestions, analyst = improve_marketing_suggestions(
            analyst, critic_message
        )
        display_suggestions(revised_suggestions)

        evaluations = evaluate_marketing_suggestions(
            topics, movie_info, revised_suggestions
        )
        display_evaluations(evaluations)

        critic_message, critic = review_marketing_suggestions(
            marketing_suggestions, critic, first_pass=False
        )
