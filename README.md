## Readme

Streamlit webapp for SocialSense product

### Important

When creating the streamlit app, specify python version = 3.10 (in advanced settings)  
Todo :

- clean requirements.txt (very heavy atm with unnecessary libraries)
- session state management : if more people were to use it, need to update session state (!)

#### Files

app.py is the main file (streamlit run app.py to launch app locally)

processing for data cleaning and LLM calls  
config for movies comparison table (7 movies with associated sentiment/emotion analysis)  
visualization for graphs

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
- Extracting the 1_000 most liked comments (maybe double check since ChatGPT but looks good)
- [not in dev] Showing neutral comments (NB : realized we currently show a general view of sentiments over all comments, not proportion of neutral vs negative vs positive)

In dev, the following is in progress :

- Using Claude instead of ChatGPT (in topic_summarization.py, both classes can be used indifferently in the functions // double check that Claude is compliant tho)
- A 'check' function (in topic_summarization.py) which is here to limit false positives when matching comments to topic, by verifying no comment is contradicting the topic it has been assigned to. Note: need to parallelize. Currently processing by batches of 30 comms.
