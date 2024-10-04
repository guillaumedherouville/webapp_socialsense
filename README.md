## Readme

Streamlit webapp for SocialSense product

### Important

If re-creating the streamlit app, specify python version = 3.10 (in advanced settings)  
Note: requirements.txt is quite heavy atm with some unnecessary libraries (not a big deal)

#### Files

main.py is the main streamlit file (cmd: streamlit run app.py to launch app locally)

from there, we can choose between dev.py and prod.py  
--> dev.py is the work in progress for development  
--> prod.py is the 'safe' page which is used for daily streamlit usage

processing for data cleaning and LLM calls  
topic_summarization for main product functions [notably, defines LLMs]
config for movies comparison table (7 movies with associated sentiment/emotion analysis)  
visualization for graphs  
app for miscellaneous  
sport for wwe-related functions

agentic.py is a work in progress to improve marketing recommendations  
streamlit_dev notebook is the notebook used to experiment it

trailer analysis v2/v3 are the initial notebooks from which the app was built  
sports analysis was a quick trial at using the product for WNBA videos / comments

excel file (marketing tactics...) is an old excel intended to help with prompting but left aside for now

#### Branches

main is for the public streamlit  
test_dev has the old streamlit (when app.py was the main file)  
devenv is for modifications before pushing to main

### State of progress

In prod, the following has been implemented :

- Topic matching fixed (i.e. the match_topic_comments function in processing.py is the same as in topic_summarization.py)
- Showing neutral comments (NB : realized we currently show a general view of sentiments over all comments, not proportion of neutral vs negative vs positive)

In dev, the following is in progress :

- Using Claude or ChatGPT (in topic_summarization.py, both classes can be used indifferently in the functions // double check that Claude is compliant tho)
- A 'check' function (in topic_summarization.py) which is here to limit false positives when matching comments to topic, by verifying no comment is contradicting the topic it has been assigned to

### Secrets

Secrets can be found in the streamlit app settings, and should be 3: [uses @guillaumedherouville YT api_key (free)]  
YT_KEY =  
OPENAI_API_KEY =  
ANTHROPIC_API_KEY =
