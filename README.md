# bdb2024

This readme was updated to reflect a clean-up of this repo. 

The core analyses of this script are found in run_analyses.ipynb and run_analyses.py. Note that the contents of these files are identical (i.e. running run_analyses.ipynb from top to bottom will yield the same results as executing run_analyses.py), but the .ipynb allows for easier manipulation of intermediate dataframes and variables. 

Prior to executing either run_analyses.* script, be sure to: 
- ensure compatibility with Python modules given in requirements.txt.
- ensure compatibility with R packages given in sessionInfo.txt.
- download and unzip the data provided from the 2024 NFL Big Data Bowl (available here: https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/data) into /data/raw_data/provided_data/. I have not included these files in the repo because of their size. 
- verify that cleaned_team_color_df.csv and plot_field.png are found in /data/raw_data/external_data/. I manually parsed through NFL team color data (available here: https://teamcolorcodes.com/nfl-team-color-codes/ -- thank you to the folks who provided this data), which comes in handy when animating plays. 
- make note of the (sub)directory structure. Running run_analyses.* should automatically create the necessary directories/subdirectories in the event that an issue has arisen. Custom subdirectory structures can be accommodated by editing SETTINGS.json.

run_analyses.* will save .csv files in /data/processed/agg_data/ and figures in /output/figs/. The script called animate_func.py has a function create_gif() which can take in a gameplayId (used commonly in these analyses as a primary key of sorts for joining and filtering) and create a gif representation of that play. These gifs will be saved in /output/gifs/plays/. Using run_analyses.ipynb is recommend for this. 

