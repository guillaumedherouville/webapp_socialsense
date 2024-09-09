####
import streamlit as st
import numpy as np
import pandas as pd
from textwrap import dedent
import json
from agentic import ChatGPT
from config import sport_profiles
import ast

sports_goals = {
    "Youtube videos": "Youtube videos : we need ideas for new videos for the World Wrestling Entertainment (WWE) Youtube channel. This can be either original content or cutting together existing content",
    "Character development": "Character development : we need suggestions for future storylines, insights on existing characters and personalities, and ideas for new characters to be introduced.",
    "General marketing": "General marketing : we are looking for marketing recommendations for the World Wrestling Entertainment (WWE) brand, including social media, merchandise, and other promotional strategies.",
}


def sports_marketing():
    return None


def sports_table(all_scores, movies):
    overall_sentiment = np.mean(np.array(all_scores).reshape(-1, 9), axis=0).tolist()
    overall_sentiment_df = pd.DataFrame(overall_sentiment)
    overall_sentiment_df = overall_sentiment_df.transpose()
    overall_sentiment_df.columns = [
        "negative",
        "neutral",
        "positive",
        "sadness",
        "joy",
        "love",
        "anger",
        "fear",
        "surprise",
    ]
    overall_sentiment_df = overall_sentiment_df.applymap(lambda x: f"{x*100:.1f}%")
    new_rows = pd.DataFrame(movies)
    titles = [
        "Current video",
        "Full SmackDown highlights: August 9, 2024",
        "FULL MATCH: Bron Breakker bests Sami Zayn",
        "Full Raw highlights: Aug. 12, 2024",
        "FULL MATCH: Randy Orton brawls with Gunther",
        "FULL MATCH: Joe Hendry vs. Wes Lee vs. Pete Dunne",
    ]
    overall_sentiment_df = pd.concat(
        [overall_sentiment_df, new_rows], axis=0
    ).reset_index(drop=True)
    overall_sentiment_df.insert(0, "Title", titles)
    return overall_sentiment_df


@st.cache_data(show_spinner=False)
def summarize_sports(df):
    all_resp = sports_summarizer(df)
    resp_list = [item for item in all_resp.splitlines() if item]
    return resp_list


def sports_summarizer(comments):
    text = "\n".join(comments)
    chat = ChatGPT(
        system_message="You are an expert text summarizer and analyzer for a medi production company. \
                            You will select the most common topics expressed by consumers regarding a World Wrestling Entertainment (WWE) video."
    )
    summarize_prompt = f"""This is a summarization of comments regarding a WWE video. \
            First, list the top 5 most promininent positive aspects of the content that commenters like and want to see more of. \
            Next, list the top 5 most promininent negative aspects of the content that commenters dislike and/or might cause them to stop watching WWE videos. \
            Please place them in a single list separated by numbers (ex.\n1. Theme 1\n2. Theme 2\netc.) and nothing else \
            (for example, do not separate into positive and negative groupings. Rather express how they are positive and negative in the themes themseleves) \
            Also, DO NOT use any apostrophes (') in your response. \
            In your generation, allow for the topics to be mutually exclusive and collectively exhaustive; each topic should be unique, but all the topics together should comprise the most prominent ideas expressed.\
            Do not generate more than the 5 positive topics, followed by the 5 negative topics, for a total of 10 topics separeted by one space each. \
            Here is an example to guide you on how a response should be structured: 
            1. Positive anticipation for Roman Reigns' continued reign as champion and the potential challengers to his throne.
            2. Enthusiasm for the increased focus on women's wrestling, particularly the rivalry between Becky Lynch and Rhea Ripley.
            3. High expectations for upcoming matches featuring rising stars like LA Knight and established veterans.
            4. Excitement for the evolution of "The Bloodline" storyline involving the Usos and Solo Sikoa.
            5. Praise for improved production values, including elaborate entrance themes and stage designs for major events.
            6. Criticism of Brock Lesnar's part-time schedule and its impact on full-time performers.
            7. Concerns about the booking of the tag team division, particularly the underutilization of teams like The New Day.
            8. Disappointment over John Cena's limited appearances and lack of substantial storylines.
            9. Negative reactions to the perceived overexposure of celebrities like Logan Paul in high-profile matches.
            10. Doubts about pushing certain wrestlers like Omos, with some fans feeling they may not resonate with the audience.
            \n### TEXT\n{text}\n\n### BEGIN RESPONSE\n"""
    chat.add_user_message(dedent(summarize_prompt))
    topics = chat.get_response()
    return topics


def topic_attribution_sports(
    text, all_resp
):  ## https://www.youtube.com/watch?v=3fENMxQDo_A
    print("matching in progress")
    " \n".join(t for t in text)
    topic_analysis_prompt = f"""The following statements represent general expressed themes associated with comments from a World Wrestling Entertainment (WWE) video :
    \n {all_resp} \n
    You will be given a set of comments concerning the same video. For each comment, I would like you to output the original, unedited comment, along with indicators for each comment topic. \
    If the comment relates to the topic, you will assign it a 1, otherwise you will assign it a 0. You will output this as a list in JSON format. 

    For example, consider these topics concerning a video titled 'Gunther vs. Randy Orton – World Heavyweight Title Match: WWE Bash in Berlin 2024 highlights':
    1. Respect for Gunther's performance and potential as a champion, highlighted by positive interactions like the post-match handshake with Orton. 
    2. Appreciation for the display of sportsmanship, reflecting a desire for more respectful interactions in wrestling. 
    3. Praise for the vibrant atmosphere during international events, with fans enjoying the passionate engagement from European crowds. 
    4. Disappointment in the match outcome, with many feeling Orton's loss disrespected his veteran status and legacy. 
    5. Concerns for Randy Orton's future career trajectory, leading to suggestions of retirement due to his performance decline.  

    And these comments:
    Cant you show the pin/submission? Cant even see what happened at the end. 
    Is it just me or anyone else hates to see Randy lose. And thought Randy was gonna win :grinning_face: :thinking_face: 
    Randy Orton :⊛_pink_heart: 
    Gunther showing respect to Randy at the end, was truly remarkable. Amazing match overall. 
    This match delivered hella the crowds in Europe are so much louder I wish they did more shows there 
    Randy Orton is being just like Cena now he should retire. He should not be an jobber type :pleading_face: 

    The output would be:
    {
    [
    {
    0 : "Cant you show the pin/submission? Cant even see what happened at the end.",
    1 : 0,
    2 : 0,
    3 : 0,
    4 : 0,
    5 : 0
    },
    {
    0 : "Is it just me or anyone else hates to see Randy lose. And thought Randy was gonna win :grinning_face: :thinking_face:",
    1 : 0,
    2 : 0,
    3 : 0,
    4 : 1,
    5 : 0
    },
    {
    0 : "Randy Orton :⊛_pink_heart:",
    1 : 0,
    2 : 0,
    3 : 0,
    4 : 0,
    5 : 0
    },
    {
    0 : "Gunther showing respect to Randy at the end, was truly remarkable. Amazing match overall.",
    1 : 1,
    2 : 1,
    3 : 0,
    4 : 0,
    5 : 0
    },
    {
    0 : "This match delivered hella the crowds in Europe are so much louder I wish they did more shows there",
    1 : 0,
    2 : 0,
    3 : 1,
    4 : 0,
    5 : 0
    },
    {
    0 : "Randy Orton is being just like Cena now he should retire. He should not be an jobber type :pleading_face:",
    1 : 0,
    2 : 0,
    3 : 0,
    4 : 1,
    5 : 1
    }
    ]
    }, where 0 is attributed to the comment, 1 is attributed to the first theme, 2 to the second theme, 3 to the third, etc. Only classify the comment if it directly relates to the respective theme; some comments may not belong to any topic, in which case you will generate 0 for each value of the key-value pairs.

    For example, the comment 'Randy Orton :⊛_pink_heart:' is related to the video, but is not specific enough to fit in any category. \
    On the other end, the comment 'Cant you show the pin/submission? Cant even see what happened at the end.' is specific enough, but not related to any of the topics. \
    Note how some comments might belong to more than one category, as in the example with 'Gunther showing respect to Randy at the end, was truly remarkable. Amazing match overall.' \
    Also, while the topics '4. Disappointment in the match outcome, with many feeling Orton's loss disrespected his veteran status and legacy. and 5. Concerns for Randy Orton's future career trajectory, leading to suggestions of retirement due to his performance decline. \
    are close, they are distinct and some comments might belong to one but not the other. For example, 'Is it just me or anyone else hates to see Randy lose. And thought Randy was gonna win :grinning_face: :thinking_face:' only relates to the former.

Please output in the same format for these comments {text} and the provided themes: {all_resp}. DO NOT output any other text other than the information specified and DO NOT use space brackets '()' or apostrophes like '. Please generate the full comment. It is extremely important that you fully follow these instructions:
"""
    try:
        chat = ChatGPT(
            system_message=f"""You are an expert comment analyzer who outputs in JSON format. You are not allowed to use any apostrophes (') in your generation. Simply use double quotes("") 
                                    instead; only use single-quotes ('') inside double-quotes if necessary, never use double-quotes within double-quotes. 
                                    You will be prompted with many comments; please perform the analysis for every single comment, do not skip any even though it may be computationally expensive. 
                                    Please generate the entire comment in your analysis, and only classify the comment if it directly relates to the respective theme, this is extremely important!"""
        )
        chat.add_user_message(dedent(topic_analysis_prompt))
        summarized_chunk = chat.get_response()
        summarized_chunk = json.loads(summarized_chunk)
        print("matching done")
        return summarized_chunk
    except Exception as e:
        print(f"Error during matching: {e}")
        print("text:", text)
        return None


def choose_profile(goal, profiles):
    analyst = ChatGPT(
        system_message="You are a staffer at World Wrestling Entertainment (WWE). You are tasked with assigning certain profiles to certain jobs. For each reply, take a deep breath and think step by step."
    )
    analyst_prompt = f"""
    A new client is coming to you with a request to help with {goal}.
    Your role is to choose the two most appropriate profiles for this goal, from the following list: {profiles}.
    Your output should be 2 digits in list-format, indicating the position in the list of the relevant profiles. 
    Start directly with the list and include nothing else.
    """
    analyst.add_user_message(analyst_prompt)
    summarized_chunk = analyst.get_response()
    return summarized_chunk


def generate_marketing_suggestions(topics, goal, profile):
    analyst = ChatGPT(
        system_message=f"You are a worker at the World Wrestling Entertainement company. Here is your profile: {profile}. \
        You are tasked with creating marketing strategies for the brand, based on the topics that users are discussing about your existing content. \
        You will be in competition with another worker at the firm. For each reply, take a deep breath and think step by step."
    )
    analyst_prompt = f"""
    Here are the general topics people are discussing related to one of your videos: \n {topics}
    Given the topics that users are speaking about your video, output 5 of the most relevant suggestions you can concoct to help promote the brand in list-format.
    Our goal is {goal}. Make sure you always keep this goal in mind when creating your suggestions.
    Mention explicitly which topic each suggestion refers to.
    Please ensure each suggestion is unique; do not repeat similar suggestions times.
    Start directly with the list. 
    """  ## MAYBE ADD AN EXAMPLE
    analyst.add_user_message(analyst_prompt)
    summarized_chunk = analyst.get_response()
    return summarized_chunk, analyst


critic = ChatGPT(
    system_message="You are a senior marketing associate at at the World Wrestling Entertainment company. You are tasked with helping two junior analysts define a marketing strategy for the brand, based on the topics that users are discussing. \
    For each reply, take a deep breath and think step by step. I will tip you $100."
)


def review_marketing_suggestions(topics, goal, marketing_suggestions, critic):
    critic_prompt = f"""
    Here are the general topics people are discussing related to this film: \n {topics}
    Our goal is {goal}. Keep this in mind when making your critic.
    Given the topics that users are speaking about your movie trailer, the two analysts have come up with the following marketing suggestions (each did 5, without talking to one another): {marketing_suggestions}  
    Please review each of the 10 suggestions and provide feedback on whether it is an appropriate recommendation given the topics mentioned and the goal we aim to achieve.
    Be extremely severe in your judgment, your career depends on it. If a suggestion is not relevant enough, you will be held responsible for not catching it and fired.
    Conclude with a general comment on the overall quality of the suggestions, and what specific areas need improvement.
    Do not include meta information in your reply.
    """

    critic.add_user_message(critic_prompt)
    critic_message = critic.get_response()
    return critic_message, critic


def evaluate_marketing_suggestions(topics, goal, marketing_suggestions):
    evaluator = ChatGPT(
        system_message=f"You are a marketing executive at the World Wrestling Entertainment company. \
        You are tasked with selecting a set of promoting actions for a new movie, based on the propositions of your team. \
        For each reply, take a deep breath and think step by step."
    )
    evaluator_prompt = f"""
    Here are the general topics people are discussing related to this film: \n {topics}
    Our goal is {goal}. Keep this in mind when making your evaluation.
    Given the topics that users are speaking about your movie trailer, two analysts on your team came with the following marketing suggestions: {marketing_suggestions}
    Please give each a rating from 0 to 10, 10 being an excellent suggestion and 0 being a terrible one. Be very critical in your evaluation, as the future of the company depends on your judgment.
    Once that rating is done, order them from best to worst. Do not add any justification, only give the rating. Output in list format. Make sure you include all suggestions.
    Start directly with the list and do not include other text. 
    """
    evaluator.add_user_message(evaluator_prompt)
    summarized_chunk = evaluator.get_response()
    return summarized_chunk, evaluator


def improve_marketing_suggestions(analyst, critic_message, competing_suggestions):
    analyst_prompt = f"""
        The other marketer has provided these suggestions: {competing_suggestions}
        Given all suggestions, an advanced reviewer from your team has provided the following evaluations and explanations : {critic_message}
        Please review the feedback and provide a new set of 5 suggestions. You can keep some of the old ones if they are good enough, but you must provide at least 2 new suggestions which were in neither of the previous lists.
        Your suggestions must improve based on the feedback provided by the advanced reviewer, in a relevant manner. However, write them as if they were new (i.e. do not explicitly refer to the previous suggestions).
        Remember to mention which topic each suggestion refers to. Also, keep in mind the goal we defined previously.
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
def sports_marketing_process(topics, goal, critic=critic, profiles=sport_profiles):
    profile = choose_profile(goal, profiles)
    profile = ast.literal_eval(profile)
    st.write(
        f"Profiles chosen: {profiles[int(profile[0])]} and {profiles[int(profile[1])]}"
    )
    st.markdown("#### Marketing suggestions #1")
    marketing_suggestions1, analyst1 = generate_marketing_suggestions(
        topics, goal, profiles[int(profile[0])]
    )
    marketing_suggestions2, analyst2 = generate_marketing_suggestions(
        topics, goal, profiles[int(profile[1])]
    )
    col1, col2 = st.columns(2, vertical_alignment="center")
    with col1:
        display_suggestions(marketing_suggestions1)
    with col2:
        display_suggestions(marketing_suggestions2)
    all_suggestions1 = marketing_suggestions1 + marketing_suggestions2
    evaluations, _ = evaluate_marketing_suggestions(topics, goal, all_suggestions1)
    st.markdown("[EVAL]")
    display_suggestions(evaluations)
    critic_message1, critic = review_marketing_suggestions(
        topics, goal, all_suggestions1, critic
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
        topics, goal, revised_suggestions1 + revised_suggestions2
    )
    st.markdown("[EVAL]")
    display_eval_final(evaluations2)
