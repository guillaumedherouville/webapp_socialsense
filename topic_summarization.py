import openai
from textwrap import dedent
import anthropic


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
        "Thank you for your response. Now, I would like you to reformat it, such that the output matches EXACTLY the example format. That is, list the 10 topics as shown and delete everything else. Remember, this is pure reformatting and it is essential you match the desired output."
    )
    topics_list = chat.get_response()
    print(topics_list)
    return topics_list


def comment_averaging(comments, context, provider, model=None):
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


def comment_voting(comments, context, provider, model=None):
    topics = comments_summarizer(comments, context, provider, model)
    system_message = (
        "You are an expert text summarizer and analyzer for a production company."
    )
    chat = (
        provider(model=model, system_message=system_message)
        if model
        else provider(system_message=system_message)
    )
    summarize_prompt = f"""You will be given lists comments regarding a movie trailer, concerning a movie with this information: \n {context} \n. \
                        Fi
                        \n### TEXT\n{text}\n\n### BEGIN RESPONSE\n"""
    chat.add_user_message(dedent(summarize_prompt))
    topics = chat.get_response()
    print(topics)
    chat.add_user_message(
        "Thank you for your response. Now, I would like you to reformat it, such that the output matches EXACTLY the example format. That is, list the 10 topics as shown and delete everything else. Remember, this is pure reformatting and it is essential you match the desired output."
    )
    topics_list = chat.get_response()
    return topics_list
