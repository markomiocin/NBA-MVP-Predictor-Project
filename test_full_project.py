"""
Marko Miocinovic
CSE 163 AE
Final Project Part 3: Code

This file Tests functionality of data_analysis.py and ml_mvp_predictor.py
files with assert statements.
plotly_graphs.py file is not tested here as graph correctness can only be
verified visually by opening the outputted HTML files
"""

import pandas as pd
import data_analysis
import ml_mvp_predictor as ml


FILE_PATH = "mvp_winners.csv"


# data_analysis.py tests

def test_winner_count(df: pd.DataFrame) -> None:
    """
    Tests that there are exactly 25 MVP winners in the dataset,
    one per season from 2000-01 through 2024-25
    """
    winner_count = len(df[df["MVP"] == 1])

    assert (winner_count == 25)


def test_non_winner_count(df: pd.DataFrame) -> None:
    """
    Tests that there are exactly 50 nonwinner rows in the dataset,
    two candidates pulled from each of the 25 seasons in the 21st century
    """
    non_winner_count = len(df[df["MVP"] == 0])

    assert (non_winner_count == 50)


def test_total_row_count(df: pd.DataFrame) -> None:
    """
    Tests that the total number of rows in the dataset is 75,
    25 winners and 50 nonwinners combined
    """
    assert (len(df) == 75)


def test_required_columns_present(df: pd.DataFrame) -> None:
    """
    Tests that all important columns are present in the cleaned dataset
    """
    required_columns = [
        "Season", "Player", "Tm", "G", "MP", "PTS", "TRB",
        "AST", "STL", "BLK", "FG%", "3P%", "FT%", "WS", "MVP"]

    for column in required_columns:
        assert (column in df.columns)


def test_dropped_columns_absent(df: pd.DataFrame) -> None:
    """
    Tests that all columns removed during cleaning
    are no longer present in the dataset
    """
    dropped_columns = ["Lg", "Voting", "Age", "WS/48", "-9999", "Start Year"]

    for column in dropped_columns:
        assert (column not in df.columns)


# ml_mvp_predictor.py tests

def test_train_model(df: pd.DataFrame) -> None:
    """
    Tests that train_model runs and returns a model and a DataFrame
    without raising any errors
    """
    model, num_stats = ml.train_model(df)
    assert (model is not None)

    # if .columns has values its len is >0, and it is a dataframe
    assert (len(num_stats.columns) > 0)


def test_num_stats_numeric_columns_only(df: pd.DataFrame) -> None:
    """
    Tests that the num_stats DataFrame returned by train_model contains
    only numeric columns, confirming non-numeric columns were all
    successfully dropped before training
    """
    model, num_stats = ml.train_model(df)
    non_numeric_columns = ["Player", "Tm", "Season", "MVP"]

    for column in non_numeric_columns:
        assert (column not in num_stats.columns)


def test_stat_importance_row_count(df: pd.DataFrame) -> None:
    """
    Tests that get_stat_importance returns one row per stat, confirming 
    every feature used in training has a resulting coefficient score
    """
    model, num_stats = ml.train_model(df)
    importance = ml.get_stat_importance(model, num_stats)

    assert (len(importance) == len(num_stats.columns))


def test_predict_mvp_winner_returns_correct_columns(
        df: pd.DataFrame) -> None:
    """
    Tests that predict_mvp_winner returns a DataFrame with a
    Player column, a MVP Probability column and no others
    """
    model, num_stats = ml.train_model(df)
    candidate_data = {
        "Player": ["Shai Gilgeous-Alexander", "Nikola Jokic"],
        "G": [53, 48],
        "MP": [33.4, 34.6],
        "PTS": [31.6, 28.8],
        "TRB": [4.4, 12.5],
        "AST": [6.4, 10.3],
        "STL": [1.4, 1.4],
        "BLK": [0.8, 0.8],
        "FG%": [0.549, 0.574],
        "3P%": [0.381, 0.391],
        "FT%": [0.895, 0.832],
        "WS": [12.1, 11.3]
    }

    candidates = pd.DataFrame(candidate_data)
    results = ml.predict_mvp_winner(candidates, model)

    assert ("Player" in results.columns)
    assert ("MVP Probability" in results.columns)
    assert (len(results.columns) == 2)


def test_predict_mvp_winner_returns_correct_row_count(
        df: pd.DataFrame) -> None:
    """
    Tests that predict_mvp_winner returns the same number of rows
    as candidates passed in, confirming no rows were dropped or duplicated
    and that all players got paired a resulting prediciton
    """
    model, num_stats = ml.train_model(df)
    candidate_data = {
        "Player": ["Shai Gilgeous-Alexander", "Nikola Jokic"],
        "G": [53, 48],
        "MP": [33.4, 34.6],
        "PTS": [31.6, 28.8],
        "TRB": [4.4, 12.5],
        "AST": [6.4, 10.3],
        "STL": [1.4, 1.4],
        "BLK": [0.8, 0.8],
        "FG%": [0.549, 0.574],
        "3P%": [0.381, 0.391],
        "FT%": [0.895, 0.832],
        "WS": [12.1, 11.3]
    }

    candidates = pd.DataFrame(candidate_data)
    results = ml.predict_mvp_winner(candidates, model)

    assert (len(results) == len(candidates))


def main():
    df = data_analysis.clean_and_filter(FILE_PATH)

    # data_analysis.py tests
    test_winner_count(df)

    test_non_winner_count(df)

    test_total_row_count(df)

    test_required_columns_present(df)

    test_dropped_columns_absent(df)

    # ml_mvp_predictor.py tests
    test_train_model(df)

    test_num_stats_numeric_columns_only(df)

    test_stat_importance_row_count(df)

    test_predict_mvp_winner_returns_correct_columns(df)

    test_predict_mvp_winner_returns_correct_row_count(df)

    print("All tests passed!")


if __name__ == "__main__":
    main()