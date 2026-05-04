# NBA MVP Predictor - CSE 163 Final Project
### Marko Miocinovic
### CSE 163 AE

## Required Libraries

Make sure the following Python libraries are installed before running the
project.

`pandas, scikit-learn, seaborn, matplotlib, plotly`


## Data

This project uses datasets sourced from [Basketball Reference](https://www.basketball-reference.com/):

* `mvp_winners.csv` — All historical NBA MVP winners with season statistics
* `Top_10_MVP_Finishers/` — A folder containing 25 CSV files, one per season
from 2000-01 through 2024-25, each holding that season's MVP award finalists

Both datasets must be present in the same directory as the Python files
for the project to run successfully. 

The folder should look something like this:

```
project/
    mvp_winners.csv
    Top_10_MVP_Finishers/
        00_01_season.csv
        01_02_season.csv
        ...
        24_25_season.csv
    data_analysis.py
    ml_mvp_predictor.py
    plotly_graphs.py
    test_full_project.py
```


## File Descriptions

* `data_analysis.py` — First it loads, cleans, and filters the MVP dataset to only
include 21st century seasons. It then merges the 2000s MVP winners with 2 non-winner candidates
from each respective NBA season in the 2000s. Finally, it performs exploratory data analysis, and 
generates two visualizations which are saved as PNG files. The data analysis and two visualizations
come from the requirements in Part 2: EDA.

* `ml_mvp_predictor.py` — Trains a logistic regression model on the dataset cleaned in `data_analysis.py`
to predict MVP probability for players. In the process of predicting probabilites, it identifies the
importance of each statistical category for predicting a MVP winner. Wraps up by predicting the worthy
winner for this 2025-26 season given a dataframe of current statistics that I manually typed into the main method.

* `plotly_graphs.py` — Uses the trained model and cleaned data to generate three interactive Plotly visualizations
saved as HTML files, each one addressing one of my research questions.

* `test_full_project.py` — Tests the core functionality of the `data_analysis.py` and `ml_mvp_predictor.py`files
using assert statements. This file does not test `plotly_graphs.py` however, as this file creates graphs
outputted as HTML files, whose correctness is verified visually by opening and accessing these files.
The `plotly_graphs.py` is built off of methods from the other two files, so testing the other two effectively
confirms a call to the graphing file will output a result.


## Instructions on How to Run

Overall, it is not complicated at all to run my project. The most important step is that all of the
files, python and csv, and in the same folder/directory. This is the only way my code will work properly
All important "running" steps are included in the code, so all it takes from a users end is to run the files
in sequence, one at a time in order to access all relevent results from all files.

**Step 1: Run the data analysis file**

`python data_analysis.py`

When doing so, the terminal will print the cleaned dataset, a summary of its dimensions and
number of missing values, and a 7-number quantitative summary of the variables of interest.
It will also save two seaborn plots to the project directory:
* `scoring_vs_minutes.png`
* `ast_vs_ws.png`

**Step 2: Run the machine learning predictor file**

`python ml_mvp_predictor.py`

Running this will print the most influential MVP statistics ranked by their logistic
regression coefficients, the predicted MVP probabilities for the 2025-26
season candidates, and a comparison showing Nikola Jokic vs Joel Embiid for
the 2022-23 season. The final comparison is a little bonus I added for my own sake.

**Step 3: Run the Plotly visualizations generating file**

`python plotly_graphs.py`

This will save three interactive HTML files to the project directory:
* `stat_importance.html` — Answers Q1: which stats best predict MVP?
* `mvp_predictions.html` — Answers Q2: predicted top 3 candidates for 2025-26
* `ws_outliers.html` — Answers Q3: which MVP seasons were Win Share outliers?

Open any of these files in a web browser to interact with the visualizations. All of the graphs represent
the same statistics/results found from running `ml_mvp_predictor.py` as the same data is manually inputted
into the main method by me already.

**Step 4: Run the functionality tests**

`python test_full_project.py`

This will run all assert based tests across `data_analysis.py` and
`ml_mvp_predictor.py` and print `All tests passed!` to the terminal 
if every assert statement is passed.


## Final Notes/Reminders on Running the Project

* All Python files must be run from the same directory as the data files
* The `Top_10_MVP_Finishers/` folder must contain all 25 season CSV files
for the project to run correctly
* Data used for MVP prediciton is manually inputted in the main method, so to try
other predictions between players or for previous seasons requires replacement of
the data, which can be found for any player and any season on [Basketball Reference](https://www.basketball-reference.com/):