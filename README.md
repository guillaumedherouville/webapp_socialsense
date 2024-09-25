## Readme

Streamlit webapp for SocialSense product

### Important

When creating the streamlit app, specify python version = 3.10 (in advanced settings)
Todo : clean requirements.txt (very heavy atm with unnecessary libraries)

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
