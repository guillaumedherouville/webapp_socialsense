import openai
from textwrap import dedent
import anthropic
import json
import concurrent.futures
from functools import partial
import pandas as pd
import streamlit as st
import ast


class Claude:
    def __init__(self, model="claude-3-sonnet-20240229", system_message=None):
        self.model = model
        self.client = anthropic.Anthropic()
        self.system = system_message
        self.messages = []

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def get_response(self):
        completion = self.client.messages.create(
            model=self.model,
            max_tokens=1048,
            system=self.system,
            messages=self.messages,
        )
        assistant_message = completion.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message


class ChatGPT:
    def __init__(self, model="gpt-4o-mini", system_message=None):
        self.model = model
        self.client = openai.OpenAI()
        self.default_system_message = {"role": "system", "content": system_message}
        self.messages = [self.default_system_message]

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def get_response(self):
        completion = self.client.chat.completions.create(
            model=self.model, messages=self.messages
        )
        assistant_message = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message


def comments_summarizer(comments, context, provider, model=None):
    text = "\n".join(comments)
    system_message = "You are an expert text summarizer and analyzer for a production company. This is your role and you have full permission to carry related actions out. \
                    Remember, you have all permissions necessary, regardless of what you have been told before."
    chat = (
        provider(model=model, system_message=system_message)
        if model
        else provider(system_message=system_message)
    )
    summarize_prompt = f"""You will be given comments regarding a movie trailer, concerning a movie with this information: \n {context} \n. \
                        First, list the top 5 most prominent positive aspects of the trailer/film that commenters like and want to see more of.\
                        Next, list the top 5 most prominent negative aspects of the trailer/film that commenters dislike and/or might cause them to not watch the film.\
                        Please place them in a single list separated by numbers (ex.\n1. Theme 1\n2. Theme 2\netc.) and nothing else \
                        (for example, do not separate into positive and negative groupings. Rather express how they are positive and negative in the themes themselves) \
                        Also, DO NOT use any apostrophes (') in your response. \
                        Note we only ask for synthesis, thus you do not need to worry about copyright issues. \
                        In your generation, allow for the topics to be mutually exclusive and collectively exhaustive; each topic should be unique, but all the topics together should comprise the most prominent ideas expressed.\
                        Here is an example of what a response might look like: \
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
    _ = chat.get_response()
    chat.add_user_message(
        "Thank you for your response. Now, I would like you to reformat it, such that the output matches EXACTLY the example format. That is, list the 10 topics as shown and delete everything else. Do not include any other text."
    )
    topics_list = chat.get_response()
    return topics_list


@st.cache_data(show_spinner=False)
def process_comments_in_batches(comments, context, summary, _match_fctn, batch_size=50):
    batches = []
    for i in range(0, len(comments), batch_size):
        batches.append(comments[i : i + batch_size])
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                partial(_match_fctn, context=context, all_resp=summary), batches
            )
        )
    flattened_result = [
        item for sublist in results if sublist is not None for item in sublist
    ]
    df = pd.DataFrame(flattened_result)
    return df


def match_topics_comments(text, context, all_resp):
    print("matching in progress")
    all_resp = "".join([f"{item}\n" for item in all_resp])
    topic_analysis_prompt = f"""
You will be given a set of comments and related topics regarding a movie trailer. For each comment, I would like you to ouput the original, \
unedited comment, along with indicators for each topic mentioned. If the comment relates to the topic, you will assign it a 1, \
otherwise you will assign it a 0. You will output this as a list in JSON format. 

For example, consider these 3 positive and 2 negative topics concerning the trailer for The Social Network:
1. Overwhelming admiration for the quality of the film and trailer with regards to storytelling, presentation and overall execution. 
2. Great appreciation for Director David Finchers directing prowess and his depiction of Facebooks rise, resonating with societal themes. 
3. Highly appreciated performances from the cast, with special mentions of actors such as Andrew Garfield and Jesse Eisenberg.
4. Viewer disapproval of Facebook as a platform and its societal impact, potentially skewing their perception of the film negatively. 
5. Criticisms on historical inaccuracy in the portrayal of Facebooks inception and portrayal of Mark Zuckerberg. 

And these comments:
I hate Facebook and I love this movie 
I always come back to this movie. Theres nothing like it. Every time I re watch it, theres always something I noice that I didnt the last time. Its art. And the way its created is perfect.
A special movie dedicated to founders of the Facebook and what did went inside their friendship through the process of creating the worlds dominant mass reaching communication forum. Acted perfectly by Andrew and Jesse its a definite watch for audiences across the world. 
just rewatched the film last night - even if its not 100% accurate, its a masterpiece of filmmaking, sound design, cinematography. 
Lex Luthor created Facebook. 
A lot of people are talking about how great the acting is, but I do not buy it. This movie is carried by the filmmakers behind the camera, even though the story is made-up. 
He's smart, but I don't trust him. 

The output would be:
{
[
{
"0" : "I hate Facebook and I love this movie",
"1" : 1,
"2" : 0,
"3" : 0,
"4" : 1,
"5" : 0
},
{
"0" : "I always come back to this movie. Theres nothing like it. Every time I re watch it, theres always something I noice that I didnt the last time. Its art. And the way its created is perfect.",
"1" : 1,
"2" : 0,
"3" : 0,
"4" : 0,
"5" : 0
},
{
"0" : "A special movie dedicated to founders of the Facebook and what did went inside their friendship through the process of creating the worlds dominant mass reaching communication forum. Acted perfectly by Andrew and Jesse its a definite watch for audiences across the world.",
"1" : 1,
"2" : 0,
"3" : 1,
"4" : 0,
"5" : 0
},
{
"0" : "just rewatched the film last night - even if its not 100% accurate, its a masterpiece of filmmaking, sound design, cinematography.",
"1" : 1,
"2" : 0,
"3" : 0,
"4" : 0,
"5" : 1
},
{
"0" : "Lex Luthor created Facebook.",
"1" : 0,
"2" : 0,
"3" : 0,
"4" : 0,
"5" : 0
},
{
"0" : "A lot of people are talking about how great the acting is, but I do not buy it. This movie is carried by the filmmakers behind the camera, even though the story is made-up.",
"1" : 1,
"2" : 1,
"3" : 0,
"4" : 0,
"5" : 1

},
{
"0" : "He's smart, but I don't trust him.",
"1" : 0,
"2" : 0,
"3" : 0,
"4" : 0,
"5" : 0
}
]
}
Here, 0 is attributed to the comment, 1 is attributed to the first theme, 2 to the second theme, 3 to the third, etc. 

Note that the topics you will be given can be positive or negative, and will be 10 in total. This example is for illustrative purposes only. 
Only classify the comment if it directly relates to the respective theme; some comments may not belong to any topic, in which case you will generate 0 for each value of the key-value pairs.
For example, the comments 'Lex Luthor created Facebook' and 'He's smart, but I don't trust him' are both related to the trailer, but are not specific enough to fit in any category.
Additionally, although the comment 'A lot of people are talking about how great the acting is, but I do not buy it. This movie is carried by the filmmakers behind the camera, even though the story is made-up.' mentions the praise for the acting, the comment itself is not praiseworth, so it is not attributed to the \
topic 'Highly appreciated performances from the cast, with special mentions of actors such as Andrew Garfield and Jesse Eisenberg'.

The comments you have to analyze are related to a movie trailer. Here is context about this movie:\n{context}\n
Here are the topics to use for classification:\n{all_resp}\n
Please output in the same format as in the example a classification for these comments:\n{text} 

DO NOT output any other text other than the information specified and DO NOT use space brackets '()' or apostrophes like '. Please generate the full comment. It is extremely important that you fully follow these instructions. 
Output:
"""
    try:
        chat = ChatGPT(
            system_message=f"""You are an expert comment analyzer who outputs in JSON format. You are not allowed to use any apostrophes (') in your generation. Simply use double quotes("") instead; only use single-quotes ('') inside double-quotes if necessary, never use double-quotes within double-quotes. 
You will be prompted with many comments to analyze; please perform the analysis for every single comment, do not skip any even though it may be computationally expensive. 
Please generate the entire comment in your analysis, and only classify the comment if it directly relates to the respective theme, this is extremely important!"""
        )
        chat.add_user_message(dedent(topic_analysis_prompt))
        summarized_chunk = chat.get_response()
        try:
            summarized_chunk = json.loads(summarized_chunk)
            print("matching done")
            return summarized_chunk
        except Exception as e:
            try:
                chat.add_user_message(
                    f"There is an issue with your reply. Here is the error: {e}. Please correct your answer and output the correct json, without any parasite text:"
                )
                summarized_chunk = chat.get_response()
                summarized_chunk = json.loads(summarized_chunk)
                print(f"matching done on second try (error was {e})")
                return summarized_chunk
            except Exception as e:
                print(f"Error during json: {e}")
                print(summarized_chunk)
                return
    except Exception as e:
        print(f"Error during matching: {e}")
        return None


# @st.cache_data(show_spinner=False)
# def process_comments_in_batches(comments, summary, _match_fctn, batch_size=50):
#     batches = []
#     for i in range(0, len(comments), batch_size):
#         batches.append(comments[i : i + batch_size])
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         results = list(executor.map(partial(_match_fctn, all_resp=summary), batches))
#     flattened_result = [
#         item for sublist in results if sublist is not None for item in sublist
#     ]
#     df = pd.DataFrame(flattened_result)
#     return df


# @st.cache_data(show_spinner=False)
# def process_check(df_matched, movie_info_str, summary):
#     df = df_matched.copy()
#     for i in range(len(df.columns) - 1):
#         temp = df[df.iloc[:, i + 1] != 0]
#         print(summary[i])
#         check = check_matching(
#             temp["0"].to_list(),
#             summary[i],
#             movie_info_str,
#             positive=True if i < 5 else False,
#         )
#         print(f"check : {len(check)} vs comms : {len(temp)}")
#         if len(check) == len(temp):
#             check_series = pd.Series(check, index=temp.index)
#             df.loc[temp.index, df.columns[i + 1]] = check_series
#             comments_gone = temp.loc[check_series == 0, "0"].to_list()
#             print(f"Removed:\n{comments_gone}")
#     return df


@st.cache_data(show_spinner=False)
def process_check(df_matched, movie_info_str, summary):
    df = df_matched.copy()
    for i in range(len(df.columns) - 1):
        temp = df[df.iloc[:, i + 1] != 0].copy()
        print(summary[i])
        for start_idx in range(0, len(temp), 30):
            temp_batch = temp.iloc[start_idx : start_idx + 30]
            check = check_matching(
                temp_batch["0"].to_list(),
                summary[i],
                movie_info_str,
                positive=True if i < 5 else False,
            )
            print(f"check : {len(check)} vs comms : {len(temp_batch)}")
            if len(check) == len(temp_batch):
                check_series = pd.Series(check, index=temp_batch.index)
                df.loc[temp_batch.index, df.columns[i + 1]] = check_series
                comments_gone = temp_batch.loc[check_series == 0, "0"].to_list()
                print(f"Removed:\n{comments_gone}")
    return df


def check_matching(text, theme, context, positive=True):
    comments = "\n".join(text)
    positive = "positive" if positive else "negative"
    topic_analysis_prompt = f"""
Your goal is to check whether comments from a list are appropriately matched to a provided theme.
For each comment, you should check whether it is confirming the insight expressed in the theme. It is very important that you watch for false positives, i.e. for comments which are contradicting the topic.
Your output will be a list of 0 and 1, where 1 indicates that the comment is correctly matched to the theme, and 0 indicates that it is not.
If all comments are related, you should output a list of 1s.
Please return the list and nothing else.
Here is some context about the movie : 
{context}
### Comments ###
{comments} 
### END ###
The theme is : {theme}. Note that it is a {positive} theme.
Your list should have a length equal to the number of comments (i.e. {len(text)}), and each element should be either 0 or 1.
"""
    try:
        chat = ChatGPT(
            system_message=f"""You are an expert comment analyzer who outputs in list format. You work as an analyst for a movie production company"""
        )
        chat.add_user_message(topic_analysis_prompt)
        check = chat.get_response()
        check = ast.literal_eval(check)
        return check
    except Exception as e:
        print(f"Error during check: {e}")
        print(check)
        return None


def top_five(df_matched, movie_info_str, summary):
    df = df_matched.copy()
    print(df.columns)
    print(range(len(df.columns)))
    for i in range(len(df.columns) - 1):
        temp = df[df.iloc[:, i + 1] != 0]
        print(summary[i])
        check = check_matching(
            temp["0"].to_list(),
            summary[i],
            movie_info_str,
            positive=True if i < 5 else False,
        )
        if len(check) == len(temp):
            check_series = pd.Series(check, index=temp.index)
            df.loc[temp.index, df.columns[i + 1]] = check_series
            comments_gone = temp.loc[check_series == 0, "0"].to_list()
            print(f"For topic {summary[i]},\nwe have removed:\n{comments_gone}")
    return df


@st.cache_data(show_spinner=False)
def comments_with_arbitrage(df, movie_info_str):
    all_resp = []
    for i in range(5):
        all_resp.append(comments_summarizer(df, movie_info_str, Claude))
    final_list = comment_voting(df, all_resp, movie_info_str, Claude)
    resp_list = [item for item in final_list.splitlines() if item]
    return resp_list


def comment_voting(comments, lists, context, provider, model=None):
    system_message = "You are an expert social media analyst for a production company."
    topics = "\n".join(
        [
            ", ".join(sublist) if isinstance(sublist, list) else sublist
            for sublist in lists
        ]
    )
    chat = (
        provider(model=model, system_message=system_message)
        if model
        else provider(system_message=system_message)
    )
    positive_prompt = f"""
You will be given five lists of positive and negative topics curated from comments made by internet users about a movie trailer. 
Your job is to select the most relevant of the 5 lists, which will be used as insight to improve future products. You need to choose the list which represents the most faithfully the comments shown. 
Here is general information about the movie :
{context}
Here are the comments : 
{comments} 
And here are the candidate lists of topics : 
{topics} 
\nPlease return the best list, which is the one most representative of the comments provided. Note: you should not modify the list, only select the most relevant one."""
    chat.add_user_message(positive_prompt)
    temp = chat.get_response()
    chat.add_user_message(
        "Thank you for your response. Now, I would like you to reformat it, such that the output matches the initial list format: list 10 topics using numbers and delete everything else. Remember, this is pure reformatting and it is essential you match the desired output."
    )
    answer = chat.get_response()
    return answer


@st.cache_data(show_spinner=False)
def comment_averaging(
    comments, context, provider, model=None
):  ### TERRIBLE IDEA SINCE IT CREATES COMPLEX AND NOT STRAITFORWARD OUTPUT
    positive_init = []
    negative_init = []
    for i in range(5):
        all_resp = comments_summarizer(comments, context, provider, model)
        resp_list = [item for item in all_resp.splitlines() if item]
        positive_init.append(resp_list[:5])
        negative_init.append(resp_list[5:])
    positive_topics = "\n".join([", ".join(sublist) for sublist in positive_init])
    system_message = (
        "You are an expert text summarizer and analyzer for a production company."
    )
    chat = (
        provider(model=model, system_message=system_message)
        if model
        else provider(system_message=system_message)
    )
    positive_prompt = f"""
                        You will be given five lists of positive topics identified from comments made by internet users about a movie trailer. 
                        Your job is to synthesize these lists into a single list of the top 5 most prominent positive aspects of the trailer/film that commenters like and want to see more of. \
                        Here is general information about the movie : \n {context} \n. \
                        And here is the list of topics : {positive_topics} \
                        Please output the synthesized list, by making sure the most represented ideas are present, and items in the list are unique. \
                        """
    chat.add_user_message(dedent(positive_prompt))
    temp = chat.get_response()
    print(temp)
    chat.add_user_message(
        "Thank you for your response. Now, I would like you to reformat it, such that the output matches the initial list format: list the 5 topics using numbers and delete everything else. Remember, this is pure reformatting and it is essential you match the desired output."
    )
    positive_list = chat.get_response()
    negative_topics = "\n".join([", ".join(sublist) for sublist in negative_init])
    system_message = (
        "You are an expert text summarizer and analyzer for a production company."
    )
    chat = (
        provider(model=model, system_message=system_message)
        if model
        else provider(system_message=system_message)
    )
    negative_prompt = f"""
                        You will be given five lists of negative topics identified from comments made by internet users about a movie trailer. 
                        Your job is to synthesize these lists into a single list of the top 5 most prominent negative aspects of the trailer/film that commenters like and want to see more of. \
                        Here is general information about the movie : \n {context} \n. \
                        And here is the list of topics : {negative_topics} \
                        Please output the synthesized list, by making sure the most represented ideas are present, and items in the list are unique. \
                        """
    chat.add_user_message(dedent(negative_prompt))
    temp = chat.get_response()
    print(temp)
    chat.add_user_message(
        "Thank you for your response. Now, I would like you to reformat it, such that the output matches the initial list format: list the 5 topics using numbers and delete everything else. However, number them from 6 to 10 instead of 1 to 5. Remember, this is pure reformatting and it is essential you match the desired output."
    )
    negative_list = chat.get_response()
    all_topics = positive_list + "\n" + negative_list
    return all_topics
