from fastapi import FastAPI
from processing import (
    generate_comments_df,
    df_character_cleaning,
    get_comments_sentiment,
    comparison_table,
)
from config import movies

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/{video_id}")
def print_top_five(video_id: str):
    df = generate_comments_df(video_id)
    df = df_character_cleaning(df)
    return df.head(5)


@app.get("/{video_id}/{movie_id}")
def results_table(video_id: str, movie_id: str):
    df = generate_comments_df(video_id)
    df = df_character_cleaning(df)
    all_scores = get_comments_sentiment(df["comment"].tolist())
    df = comparison_table(all_scores, movie_id, movies)
    return df.to_dict()
