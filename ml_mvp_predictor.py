"""
Marko Miocinovic
CSE 163 AE
Final Project Part 3: Code

This file runs a logistic regression machine learning
model to predict which player from a given group of
statistics is most likely to win the MVP, using past MVP
winning and runnner up data.
"""

import pandas as pd
import data_analysis # gives this file access to cleaned and filtered data
from sklearn.linear_model import LogisticRegression

CANDIDATE_DATA = {
                "Player": [
                    "Shai Gilgeous-Alexander", "Nikola Jokic",
                    "Cade Cunningham", "Victor Wembanyama", "Jaylen Brown"],
                "G": [53, 48, 56, 50, 57],
                "MP": [33.4, 34.6, 35.2, 29.1, 34.3],
                "PTS": [31.6, 28.8, 25.2, 23.9, 28.7],
                "TRB": [4.4, 12.5, 5.7, 11.1, 7.2],
                "AST": [6.4, 10.3, 9.8, 2.9, 5.1],
                "STL": [1.4, 1.4, 1.4, 1.0, 1.0],
                "BLK": [0.8, 0.8, 0.9, 3.0, 0.4],
                "FG%": [0.549, 0.574, 0.456, 0.505, 0.480],
                "3P%": [0.381, 0.391, 0.343, 0.350, 0.348],
                "FT%": [0.895, 0.832, 0.813, 0.817, 0.780],
                "WS": [12.1, 11.3, 7.0, 7.2, 5.6]
            }


def train_model(df: pd.DataFrame) -> tuple[LogisticRegression, pd.DataFrame]:
    """
    Trains a logistic regression model on historical MVP data, returning the
    fitted model as well as the dataframe of numerical stats that was used
    to train the model
    """
    # keep only numeric stats, points, rebounds, etc.
    X = df.drop(columns=["Player", "Tm", "MVP", "Season"])
    y = df["MVP"]

    # terminal recommends increase of max_iter "to improve the convergence"
    model = LogisticRegression(max_iter=1000)

    # Train model, given the stats (X) and the resulting MVP status 0 or 1 (y)
    model.fit(X, y)

    return (model, X)


def get_stat_importance(
    model: LogisticRegression, num_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame of each relevant numerical statistic and its
    resulting logistic regression coefficient sorted in descending order,
    aka its "importance" in deciding a players MVP status
    """
    # using [0] to turn to a list, since .coef_ returns a 
    # "ndarray of shape (1, n_features)"
    importance = pd.DataFrame(
            {"Statistic": num_stats.columns, "Importance": model.coef_[0]})

    importance = importance.sort_values(by="Importance", ascending=False)
    
    return importance


def predict_mvp_winner(
    candidates: pd.DataFrame, model: LogisticRegression) -> pd.DataFrame:
    """
    Uses trained logistic regression model to predict probability of each
    candidate winning the MVP. Returns a dataframe holding all player names
    and the probability of them winning the MVP, sorting probabilities in
    descending order
    """
    # remove name column so only numerical remain for prediction
    candidate_stats = candidates.drop(columns=["Player"])

    # Predict probability of winning MVP for all candidates. Slicing to
    # keep only all rows of second column, since first column holds
    # probability of not winning and second holds probability of winning
    probs = model.predict_proba(candidate_stats)
    prob_winning = probs[:, 1]

    # Add probabilities of winning to candidate df
    candidates["MVP Probability"] = prob_winning

    candidates = candidates.sort_values(by="MVP Probability", ascending=False)

    # only return the player name and their likelihood of winning MVP
    return candidates[["Player", "MVP Probability"]]


def main():
    df = data_analysis.clean_and_filter(data_analysis.WINNERS_FILE)

    # training the model, unpack returned tuple so variables can be used
    # individually as parameters in other methods
    model, num_stats = train_model(df)

    ## Showing stats that best predict MVP winners ##
    importance = get_stat_importance(model, num_stats)
    print("Most Influential MVP Statistics")
    print(importance)


    ## Predicting MVP winner 25-26 season ##

    # all of CANDIDATE_DATA found on basketball reference player 
    # profile of 25-26 season averages on 03/09/2026
    candidates = pd.DataFrame(CANDIDATE_DATA)

    results = predict_mvp_winner(candidates, model)
    print("\nMVP Prediction Results 25/26 Season:")
    print(results)


    ## Proof Jokic was Robbed 22-23! ##
    robbery_data = {
        "Player": ["Nikola Jokic", "Joel Embiid"],
        "G": [69, 66],
        "MP": [33.7, 34.6],
        "PTS": [24.5, 33.1],
        "TRB": [11.8, 10.2],
        "AST": [9.8, 4.2],
        "STL": [1.3, 1.0],
        "BLK": [0.7, 1.7],
        "FG%": [0.632, 0.548],
        "3P%": [0.383, 0.330],
        "FT%": [0.822, 0.857],
        "WS": [14.9, 12.3]
    }

    robbery = pd.DataFrame(robbery_data)

    # second call to method, do not print importance of stats
    robbery_results = predict_mvp_winner(robbery, model)

    print("\nProof Jokic Deserved 22/23 Season MVP:")
    print(robbery_results)


if __name__ == '__main__':
    main()