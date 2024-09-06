####
import streamlit as st
import numpy as np
import pandas as pd
from processing import ChatGPT, chunkify_by_tokens, num_tokens_from_string


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
    all_resp = SPORTS_SUMMARIZER(df, 3900)
    resp_list = [item for item in all_resp.splitlines() if item]
    return resp_list


def SPORTS_SUMMARIZER(texts, token_threshold):
    text = "\n".join(texts)
    chunks = chunkify_by_tokens(text, token_threshold)

    first_pass_summaries = []

    for ci, chunk in enumerate(chunks):
        summarize_prompt = f"Please summarize the following set of comments: \
                            \n\n### COMMENT TEXT\n{chunk}\n\n### BEGIN RESPONSE\n"
        try:
            chat = ChatGPT(
                system_message="You are an expert text summarizer and analyzer."
            )
            chat.add_user_message(summarize_prompt)
            summarized_chunk = chat.get_response()
            first_pass_summaries.append(summarized_chunk)

        except Exception as e:
            print(f"Error during initial summarization: {e}")
            # Continue to next chunk instead of returning
            continue

    candidate_text = "\n".join(first_pass_summaries)

    iterations = 0
    candidate_len = np.inf

    while candidate_len > token_threshold:
        iterations += 1
        chunks = chunkify_by_tokens(candidate_text, token_threshold)

        summarized_chunks = []
        for chunk in chunks:
            try:
                chat = ChatGPT(
                    system_message="You are an expert text summarizer and analyzer. You are recursively summarizing topics expressed by consumers to select the most commonly expressed topics."
                )
                summarize_prompt = f"This is a summarization of comments regarding a World Wrestling Entertainment (WWE) video. \
                                    List the top 5 most promininent positive aspects of the content that commenters like and want to see more of.\
                                    List the top 5 most promininent negative aspects of the content that commenters dislike and/or might cause them to stop watching WWE videos.\
                                    Be specific in your generation of topics, and ensure that each topic is distinct. \
                                    \n### TEXT\n{chunk}\n\n### BEGIN RESPONSE\n"
                chat.add_user_message(summarize_prompt)
                summarized_chunk = chat.get_response()
                # Verify and process the summarized chunk here if necessary
                summarized_chunks.append(summarized_chunk)
            except Exception as e:
                print(f"Error during iterative summarization: {e}")
                # Optionally handle the error, like retrying summarization for this chunk
                continue

        candidate_text = " ".join(summarized_chunks)
        candidate_len = num_tokens_from_string(candidate_text)

    try:
        chat = ChatGPT(
            system_message="You are an expert text summarizer and analyzer for a medi production company.  \
                                    You will select the most common topics expressed by consumers regarding a World Wrestling Entertainment (WWE) video."
        )
        summarize_prompt = f"""This is a summarization of comments regarding a WWE video. \
                            First, list the top 5 most promininent positive aspects of the content that commenters like and want to see more of.\
                            Next, list the top 5 most promininent negative aspects of the content that commenters dislike and/or might cause them to stop watching WWE videos.\
                            Please place them in a single list separated by by numbers (ex. 1. Theme 1 \n 2. Theme 2 \n etc.) and nothing else \
                            (for example, do not separate into positive and negative groupings. Rather express how they are positive and negative in the themes themseleves) \
                            Also, DO NOT use any apostrophes (') in your response. \
                            In your generation, allow for the topics to be mutually exclusive and collectively exhaustive; each topic should be unique, but all the topics together should comprise the most prominent ideas expressed.\
                            Do not generate more than the 5 positive topics, followed by the 5 negative topics, for a total of 10 topics separeted by one space each. \
                            Here is an example to guide you on how a response should be structured: \
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
                            \n### TEXT\n{candidate_text}\n\n### BEGIN RESPONSE\n"""
        chat.add_user_message(summarize_prompt)
        final_response = chat.get_response()
        # Verify and process the summarized chunk here if necessary
    except Exception as e:
        print(f"Error during iterative summarization: {e}")
        # Optionally handle the error, like retrying summarization for this chunk
        return candidate_text

    return final_response


def topic_attribution_sports(text, all_resp):
    print("matching in progress")
    " \n".join(t for t in text)
    topic_analysis_prompt = f"""The following statements represent general expressed themes associated with a set of movie trailer comments \n {all_resp} \n
  You will be given a set of comments concerning the same movie trailer. For each comment, I would like you
  to ouput the original, unedited comment, along with indicators for each comment topic. If the comment relates to the topic,
  you will assign it a 1, otherwise you will assign it a 0. You will output this as a list in JSON format. \n

  For example, consider these topics concerning the trailer for The Social Network:
  1. Overwhelming admiration for the quality of the film and trailer with regards to storytelling, presentation and overall execution. \n
  2. Great appreciation for Director David Finchers directing prowess and his depiction of Facebooks rise, resonating with societal themes. \n
  3. Highly appreciated performances from the cast, with special mentions of actors such as Andrew Garfield and Jesse Eisenberg.
  4. Viewer disapproval of Facebook as a platform and its societal impact, potentially skewing their perception of the film negatively. \n
  5. Criticisms on historical inaccuracy in the portrayal of Facebooks inception and portrayal of Mark Zuckerberg. \n
    1. Praise for Drew McIntyre's standout performances and engaging storylines, particularly his rivalry with CM Punk. \n
    2. Appreciation for the inclusion of humor and lighthearted moments, enhancing the enjoyable aspects of the programming. \n
    3. Interest in evolving character developments like those of Rhea Ripley and Damian Priest, with a desire for deeper portrayals. \n
    4. Negative comparisons to AEW, with viewers seeking greater excitement and narrative depth in WWE programming. \n
    5. Frustration with the repetitiveness of storylines, indicating a need for innovative narratives to maintain audience interest. \n


  And these comments:
  I hate Facebook and I love this movie \n
  I always come back to this movie. Theres nothing like it. Every time I re watch it, theres always something I noice that I didnt the last time. Its art. And the way its created is perfect. \n
  A special movie dedicated to founders of the Facebook and what did went inside their friendship through the process of creating the worlds dominant mass reaching communication forum. Acted perfectly by Andrew and Jesse its a definite watch for audiences across the world. \n
  just rewatched the film last night - even if its not 100% accurate, its a masterpiece of filmmaking, sound design, cinematography. \n
  Lex Luthor created Facebook. \n
  A lot of people are talking about how great the acting is, but I do not buy it. This movie is carried by the filmmakers behind the camera, even though the story is made-up. \n
  He's smart, but I don't trust him. \n


  The output would be:
    {
    [
    {
  0 : "I hate Facebook and I love this movie",
  1 : 1,
  2 : 0,
  3 : 0,
  4 : 1,
  5 : 0
    },
    {
  0 : "I always come back to this movie. Theres nothing like it. Every time I re watch it, theres always something I noice that I didnt the last time. Its art. And the way its created is perfect.",
  1 : 1,
  2 : 0,
  3 : 0,
  4 : 0,
  5 : 0
    },
    {
  0 : "A special movie dedicated to founders of the Facebook and what did went inside their friendship through the process of creating the worlds dominant mass reaching communication forum. Acted perfectly by Andrew and Jesse its a definite watch for audiences across the world.",
  1 : 1,
  2 : 0,
  3 : 1,
  4 : 0,
  5 : 0
    },
    {
  0 : "just rewatched the film last night - even if its not 100% accurate, its a masterpiece of filmmaking, sound design, cinematography.",
  1 : 1,
  2 : 0,
  3 : 0,
  4 : 0,
  5 : 1
    },
    {
  0 : "Lex Luthor created Facebook.",
  1 : 0,
  2 : 0,
  3 : 0,
  4 : 0,
  5 : 0
    },
    {
  0 : "A lot of people are talking about how great the acting is, but I do not buy it. This movie is carried by the filmmakers behind the camera, even though the story is made-up.",
  1 : 1,
  2 : 1,
  3 : 0,
  4 : 0,
  5 : 1

    },
    {
  0 : "He's smart, but I don't trust him.",
  1 : 0,
  2 : 0,
  3 : 0,
  4 : 0,
  5 : 0
    }
    ]
    }, where 0 is attributed to the comment, 1 is attributed to the first theme, 2 to the second theme, 3 to the third, etc. Only classify the comment if it directly relates to the respective theme; some comments may not belong to any topic, in which case you will generate 0 for each value of the key-value pairs.

    For example, the comments 'Lex Luthor created Facebook' and 'He's smart, but I don't trust him' are both related to the trailer, but are not specific enough to fit in any category.

    Additionally, although the comment 'A lot of people are talking about how great the acting is, but I do not buy it. This movie is carried by the filmmakers behind the camera, even though the story is made-up.' mentions the praise for the acting, the comment itself is not praiseworth, so it is not attributed to the
    topic 'Highly appreciated performances from the cast, with special mentions of actors such as Andrew Garfield and Jesse Eisenberg'.
.
Please output in the same format for these comments {text} and the provided themes: {all_resp}. DO NOT output any other text other than the information specified and DO NOT use space brackets '()' or apostrophes like '. Please generate the full comment. It is extremely important that you fully follow these instructions:
"""
    try:
        chat = ChatGPT(
            system_message=f"""You are an expert comment analyzer who outputs in JSON format. You are not allowed to use any apostrophes (') in your generation. Simply use double quotes("") 
                                    instead; only use single-quotes ('') inside double-quotes if necessary, never use double-quotes within double-quotes. 
                                    You will be prompted with many comments; please perform the analsys for every single comment, do not skip any even though it may be computationally expensive. 
                                    Please generate the entire comment in your analysis, and only classify the comment if it directly relates to the respective theme, this is extremely important!"""
        )
        chat.add_user_message(topic_analysis_prompt)
        summarized_chunk = chat.get_response()
        summarized_chunk = json.loads(summarized_chunk)
        print("matching done")
        return summarized_chunk
    except Exception as e:
        print(f"Error during matching: {e}")
        print("text:", text)
        return None
    



@st.cache_data(show_spinner=False)
def sports_marketing(resp_list):
    topic_analysis_prompt = f"""
    You are tasked with providing marketing suggestions based on the topics that users are discussing related to a World Wrestling Entertainment (WWE) video.

    Here are the general topics people are discussing related to this video: \n {resp_list}

    Given the topics that users are speaking about this content, output 5 of the most relevant marketing suggestions you can concoct to help promote the film in list-format.

    Please lend creative and specific suggestions to help market similar content.

    Please ensure each suggestion is unique; do not repeat similar suggestions multiple times.

    Start directly with the list, do not include other text, and be concise (yet detailed) in your suggestions."""
    try:
        chat = ChatGPT(
            system_message="You are an expert marketing analyzer who outputs marketing advice given topics that users are speaking about. Your client is the World Wrestling Entertainment (WWE) production company."
        )
        chat.add_user_message(topic_analysis_prompt)
        summarized_chunk = chat.get_response()

    except Exception as e:
        print(f"Error during initial summarization: {e}")

    return summarized_chunk
