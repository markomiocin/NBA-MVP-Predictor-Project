"""
Marko Miocinovic
CSE 163 AE
Final Project Part 3: Code

This file plots out visualization to answer the three research question
that were posed at the start of the project process
"""

import pandas as pd
import plotly.express as px
import data_analysis
import ml_mvp_predictor as ml


def plot_stat_importance(importance: pd.DataFrame) -> None:
    """
    Answers Q1: Which specific player statistics are the best at predicting
    the rightful winner of the NBA MVP award?

    Displays a bar chart of each statistic's logistic regression
    coefficient
    """
    fig = px.bar(
        importance,
        x="Importance",
        y="Statistic",
        title="Importance of Relevant Stats in Predicting MVP (2000-2025)")

    fig.write_html("stat_importance.html")


def plot_mvp_predictions(results: pd.DataFrame) -> None:
    """
    Answers Q2: Can I predict the top three MVP candidates for a given
    NBA season?

    Displays a bar chart of each MVP candidate's predicted MVP probability
    for the 2025-26 season
    """
    fig = px.scatter(
        results,
        x="Player",
        y="MVP Probability",
        title="Predicted Final MVP Candidate Standings (2025/26 NBA Season)")

    fig.update_traces(marker_size=20)

    fig.write_html("mvp_predictions.html")


def plot_ws_outliers(df: pd.DataFrame) -> None:
    """
    Answers Q3: Which years of MVP winners were statistical outliers
    compared to the usual MVP trends?

    Displays a bar chart of Win Shares for each MVP winner, making it
    easy to spot unusually high or low win share seasons
    """
    # from the full dataset look at only MVP winners (MVP == 1)
    # terminal gave SettingWithCopyWarning if .copy() is not used
    winners = df[df["MVP"] == 1].copy()

    # create new column to hold season-unique names so repeated names are
    # not stacked in the bar plot and each MVP season has its own bar
    winners["Player Season"] = winners["Player"] + " " + winners["Season"]

    fig = px.bar(
        winners,
        x="Player Season",
        y="WS",
        hover_data=["Season", "Tm"],
        title="Win Shares of NBA MVP Winners (2000-2025)")

    fig.write_html("ws_outliers.html")


def main():
    df = data_analysis.clean_and_filter(data_analysis.WINNERS_FILE)

    # train once for overall use, unpack returned tuple so variables can
    # be used individually as parameters in other methods
    model, num_stats = ml.train_model(df)

    ## Which stats best predict MVP? ##
    importance = ml.get_stat_importance(model, num_stats)
    plot_stat_importance(importance)
    print("Stat Importance Graph Plotted Succesfully!\n")

    ## Predict top 3 MVP candidates for 25-26 season ##

    # data found in costant variable at top of ml_mvp_predictor.py
    candidates = pd.DataFrame(ml.CANDIDATE_DATA)

    results = ml.predict_mvp_winner(candidates, model)
    plot_mvp_predictions(results)
    print("MVP Predictions Graph Plotted Succesfully!\n")

    ## Q3: Which MVP years were Win Share outliers? ##
    plot_ws_outliers(df)
    print("MVP Outliers Graph Plotted Succesfully!\n")


if __name__ == '__main__':
    main()