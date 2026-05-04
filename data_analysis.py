"""
Marko Miocinovic
CSE 163 AE
Final Project Part 3: Code

This file cleans a dataset containing stats and information about all 
historical MVP winnders of the NBA, filtering the original data frame to
only 21st century winners, and performs an overall exploratory data analysis
of the newly cleaned data.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# main dataset including all historical winners
WINNERS_FILE = "mvp_winners.csv"

# MVP finalists per season, ranging top 9 to top 17 depending on year
CONTENDERS_FILES = [
    "Top_10_MVP_Finishers/24_25_season.csv",
    "Top_10_MVP_Finishers/23_24_season.csv",
    "Top_10_MVP_Finishers/22_23_season.csv",
    "Top_10_MVP_Finishers/21_22_season.csv",
    "Top_10_MVP_Finishers/20_21_season.csv",
    "Top_10_MVP_Finishers/19_20_season.csv",
    "Top_10_MVP_Finishers/18_19_season.csv",
    "Top_10_MVP_Finishers/17_18_season.csv",
    "Top_10_MVP_Finishers/16_17_season.csv",
    "Top_10_MVP_Finishers/15_16_season.csv",
    "Top_10_MVP_Finishers/14_15_season.csv",
    "Top_10_MVP_Finishers/13_14_season.csv",
    "Top_10_MVP_Finishers/12_13_season.csv",
    "Top_10_MVP_Finishers/11_12_season.csv",
    "Top_10_MVP_Finishers/10_11_season.csv",
    "Top_10_MVP_Finishers/09_10_season.csv",
    "Top_10_MVP_Finishers/08_09_season.csv",
    "Top_10_MVP_Finishers/07_08_season.csv",
    "Top_10_MVP_Finishers/06_07_season.csv",
    "Top_10_MVP_Finishers/05_06_season.csv",
    "Top_10_MVP_Finishers/04_05_season.csv",
    "Top_10_MVP_Finishers/03_04_season.csv",
    "Top_10_MVP_Finishers/02_03_season.csv",
    "Top_10_MVP_Finishers/01_02_season.csv",
    "Top_10_MVP_Finishers/00_01_season.csv"]


def add_non_winners(main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loops through each 2000s' seasons top MVP candidates, extracts the
    runner-up and 8th place MVP candidates, adding a MVP row set = 0, and
    adds them to the main dataset of all historical winners
    """

    non_winners = pd.DataFrame()

    for file in CONTENDERS_FILES:
        df = pd.read_csv(file)

        # keeps track of these players as non-MVP winners though first row
        # player was technically a winner, this is irrelevant as they are not
        # being transfered and are included already in the winners only
        # master csv file
        df["MVP"] = 0

        # choose the year's runner up and 8th place candidates,
        # keeping only relevant columns
        selected_rows = df.loc[[1, 7], [
            "Player", "Tm", "G", "MP", "PTS", "TRB", "AST",
            "STL", "BLK", "FG%", "3P%", "FT%", "WS", "MVP"]]

        non_winners = pd.concat([non_winners, selected_rows],
                                ignore_index=True)

    # combine with file containing only MVP winners
    combined_df = pd.concat([main_df, non_winners], ignore_index=True)

    return combined_df


def clean_and_filter(filepath: str) -> pd.DataFrame:
    """
    Loads the MVP CSV (skipping the descriptive header row), drops 
    irrelevant columns, and filters the dataset to only include 
    MVP winnters in the 21st century.
    """
    df = pd.read_csv(filepath)

    # Create numeric "start year" column for filtering
    df["Start Year"] = df["Season"].str[:4].astype(int)
    
    # Filter for the 21st Century
    df_21st_century = df[df["Start Year"] >= 2000]

    # Drop unnecessary columns, no more need for Start Year already used,
    # only using winshares, win shares per 48 repetitive
    cols_to_drop = ["Lg", "Voting", "Age", "WS/48", "-9999", "Start Year"]
    df_21st_century = df_21st_century.drop(columns=cols_to_drop)

    # Add MVP column: binary representation of winner or not, all
    # set to 1 for now as current data state holds only past winners
    df_21st_century["MVP"] = 1

    # add non-mvp winning candidates
    full_df_21st_century = add_non_winners(df_21st_century)
    
    return full_df_21st_century


def summarize_data(df: pd.DataFrame) -> None:
    """
    Prints all info for EDA results, including dataset dimensions,
    a missing values check, and summary of variables of interest.
    """
    print("Modern Era MVP Dataset Summary:")

    row_count = len(df)
    col_count = len(df.columns)
    print(f"Size: {row_count} rows by {col_count} columns")
    
    # Check for missing data, first .sum() returns series of missing values 
    # per column, second .sum() adds up all columns in the new series for a
    # overall missing value count
    total_missing = df.isnull().sum().sum()
    print(f"Total Missing Values: {total_missing}\n")
    
    # Quantitative Summary
    print("Quantitative (7-number) Summary:")
    vars_of_interest = ["PTS", "TRB", "AST", "STL", "BLK", "WS"]
    print(df[vars_of_interest].describe())


def dataset_visualizations(df: pd.DataFrame) -> None:
    """
    Generates visualizations for scoring trends over time and
    the relationship between assists and win shares.
    """
    sns.set_theme()

    # Plot 1: Points Per Game vs Year
    sns.relplot(data=df, x="MP", y="PTS", kind="line", marker="o", aspect=2)

    plt.title("Correlation between Points and Minutes per game (2000-2025)")
    plt.xlabel("Minutes Per Game (MP)")
    plt.ylabel("Points Per Game (PTS)")
    plt.savefig("scoring_vs_minutes.png", bbox_inches="tight")

    # Plot 2: Assists per Game vs Win Shares
    sns.catplot(data=df, x="AST", y="WS", kind="bar", aspect=3)

    plt.title("Impact of Playmaking (Assists) on Team Winning (WS) (2000-25)")
    plt.xlabel("Assists Per Game (AST)")
    plt.ylabel("Total Win Shares (WS)")
    plt.xticks(rotation=-90)
    plt.savefig("ast_vs_ws.png", bbox_inches="tight")


def main():
    df = clean_and_filter(WINNERS_FILE)
    print(df)
    summarize_data(df)
    dataset_visualizations(df)

if __name__ == '__main__':
    main()
