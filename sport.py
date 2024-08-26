####

THE GOAL IS TO HAVE ALL GEN FUNCTIPNS HERE, WHEN THEY RE DIFF FOR SPORT VS MOVIE

####

def comparison_table(all_scores, movie_id, movies):
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

    ia = IMDb()
    movie = ia.get_movie(movie_id)

    new_rows = pd.DataFrame(movies)
    titles = [
        movie.get("title"),
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
