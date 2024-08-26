import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import streamlit as st


def percent_to_float(s):
    return float(s.strip("%")) / 100


def sentiment_viz(overall_sentiment_df):
    # Pre-processing
    sentiment_df = overall_sentiment_df[
        ["Title", "negative", "neutral", "positive"]
    ].copy()
    for col in sentiment_df.columns[1:]:
        sentiment_df[col] = sentiment_df[col].apply(percent_to_float)

    # Sentiment Vizualization
    fig, ax = plt.subplots(figsize=(18, 10))
    n_movies = len(sentiment_df["Title"])
    width = 0.12
    spacing = 0.02
    colors = plt.cm.viridis(np.linspace(0, 1, n_movies))
    for idx, movie in enumerate(sentiment_df["Title"]):
        plt.bar(
            [p + idx * (width + spacing) for p in range(len(sentiment_df.columns[1:]))],
            sentiment_df.loc[idx, sentiment_df.columns[1:]],
            width=width,
            label=movie,
            color=colors[idx],
        )
    ax.set_xticks(
        [
            p + (width + spacing) * n_movies / 2
            for p in range(len(sentiment_df.columns[1:]))
        ]
    )
    ax.set_xticklabels(sentiment_df.columns[1:], fontsize=14)
    legend = plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, fontsize=12)
    legend.set_title("Comparisons", prop={"size": 14})  # Set legend title
    legend.get_frame().set_facecolor("white")  # Set legend background color
    legend.get_frame().set_edgecolor("black")
    plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment="right")
    ax.set_title(
        "Comparison of Sentiment for Other References",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Sentiments", fontsize=16)
    ax.set_ylabel("Percentage", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)


def emotion_viz(overall_sentiment_df):
    # Pre-processing
    emotion_df = overall_sentiment_df[
        ["Title", "sadness", "joy", "love", "anger", "fear", "surprise"]
    ].copy()
    for col in emotion_df.columns[1:]:
        emotion_df[col] = emotion_df[col].apply(percent_to_float)

    # Emotion Vizualization
    fig, ax = plt.subplots(figsize=(18, 10))
    n_movies = len(emotion_df["Title"])
    width = 0.12
    spacing = 0.02
    colors = plt.cm.viridis(np.linspace(0, 1, n_movies))
    for idx, movie in enumerate(emotion_df["Title"]):
        plt.bar(
            [p + idx * (width + spacing) for p in range(len(emotion_df.columns[1:]))],
            emotion_df.loc[idx, emotion_df.columns[1:]],
            width=width,
            label=movie,
            color=colors[idx],
        )
    ax.set_xticks(
        [
            p + (width + spacing) * n_movies / 2
            for p in range(len(emotion_df.columns[1:]))
        ]
    )
    ax.set_xticklabels(emotion_df.columns[1:], fontsize=14)
    legend = plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, fontsize=12)
    legend.set_title("Comparisons", prop={"size": 14})
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment="right")
    ax.set_title(
        "Comparison of Emotion for Other References",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Emotions", fontsize=16)
    ax.set_ylabel("Percentage", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)


def display_comments_by_topic(df):
    temp = df.fillna(0)
    # Sum the counts across rows to get total counts for each topic
    df_counts = temp.sum(axis=0).reset_index()
    df_counts.columns = ["Topic", "Count"]

    # Ensure all possible topics (1-10) are included by filling in missing topics with a count of 0
    all_topics = [str(i) for i in range(1, 11)]  # Topics numbered 1-10 as strings
    df_counts = df_counts.set_index("Topic")
    df_counts = df_counts.reindex(all_topics, fill_value=0).reset_index()

    # Convert 'Topic' column back to integer if needed
    df_counts["Topic"] = df_counts["Topic"].astype(int)

    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    # Define your color palette
    muted_green = "#6dbf67"
    coolwarm_red = "#d73027"
    palette = [muted_green] * 5 + [
        coolwarm_red
    ] * 5  # First 5 bars in muted_green, next 5 in coolwarm_red

    # Create the count plot
    ax = sns.barplot(data=df_counts, x="Topic", y="Count", palette=palette)

    # Set title and labels with a lighter color for visibility
    plt.title("Number of Comments by Topic")  # , color="white")
    plt.xlabel("Topic")  # , color="white")
    plt.ylabel("Count of Comments")  # , color="white")

    # Adding the count above each bar
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=10,
            xytext=(0, 5),
            textcoords="offset points",
        )

    # Calculate total counts for the first 5 and next 5 topics
    first_5_total = df_counts[df_counts["Topic"].isin(df_counts["Topic"].unique()[:5])][
        "Count"
    ].sum()
    next_5_total = df_counts[df_counts["Topic"].isin(df_counts["Topic"].unique()[5:])][
        "Count"
    ].sum()

    # Get current axes and calculate percentages
    ax = plt.gca()
    positive_perc = first_5_total / (first_5_total + next_5_total) * 100
    negative_perc = next_5_total / (first_5_total + next_5_total) * 100

    # Infographic text
    plt.text(
        7.8,
        ax.get_ylim()[1] * 0.92,
        f"POS : {positive_perc:.4g}%\nNEG : {negative_perc:.4g}%",
        fontweight="bold",
    )

    # Create the plot
    st.pyplot(fig)
