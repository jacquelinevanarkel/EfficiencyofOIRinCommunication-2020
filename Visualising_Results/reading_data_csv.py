# Necessary imports
import pandas as pd
from pathlib import Path
import numpy as np
import pickle

# ---------------------------------- Reading in data & preprocessing before plotting -----------------------------------
# Specify the directory in which you stored the results
entries = Path('/Users/Jacqueline/Documents/Research_Internship/Implementation/Simulations/Simulation2/Simulation2')

# Initialize pandas dataframe to store results
results = pd.DataFrame(
    columns=['Intention Speaker', 'Inferred Referent Listener', 'Number of Turns', 'Order of Reasoning Speaker',
             'Order of Reasoning Listener', 'Communicative Success', 'Reached Threshold Order',
             'Reached Threshold Interaction', 'Dialogue History', 'Entropy', 'Probability Dist. Given Signal',
             'Ambiguity Level', 'Number of Signals', 'Number of Referents', 'Entropy Threshold'])

# Read in all files of the directory in which you stored the results and concatenate them in order to visualise
for entry in entries.iterdir():
    if str(entry).startswith('/Users/Jacqueline/Documents/Research_Internship/Implementation/Simulations/Simulation2/'
                             'Simulation2/amb'):
        result = pd.read_pickle(entry)
        results = pd.concat([results, result])

# Replace the entropy threshold for pragmatic agents with order 2 with NaN as it's not applicable
results.loc[results["Order of Reasoning Listener"] == 2, "Entropy Threshold"] = np.nan

# Take the results of the interactional and pragmatic agents separately
results_interactional = results[results["Order of Reasoning Listener"] == 0]
results_pragmatic = results[results["Order of Reasoning Listener"] != 0]

# Different pragmatic agents
results_1 = results_pragmatic[(results_pragmatic["Order of Reasoning Listener"] == 1) & (results_pragmatic["Reached Threshold Order"]==False)]
results_2 = results_pragmatic[results_pragmatic["Order of Reasoning Listener"] == 2]
results_frugal = results_pragmatic[(results_pragmatic["Order of Reasoning Listener"] == 1) & (results_pragmatic["Reached Threshold Order"]==True)]

# Store the results to be used later for plotting
compression_opts = dict(method='zip',
                        archive_name='results.csv')
results.to_csv('results.zip', index=False,
          compression=compression_opts)

compression_opts = dict(method='zip',
                        archive_name='results_interactional.csv')
results_interactional.to_csv('results_interactional.zip', index=False,
          compression=compression_opts)

compression_opts = dict(method='zip',
                        archive_name='results_pragmatic.csv')
results_pragmatic.to_csv('results_pragmatic.zip', index=False,
          compression=compression_opts)

compression_opts = dict(method='zip',
                        archive_name='results_1.csv')
results_1.to_csv('results_1.zip', index=False,
          compression=compression_opts)

compression_opts = dict(method='zip',
                        archive_name='results_2.csv')
results_2.to_csv('results_2.zip', index=False,
          compression=compression_opts)

compression_opts = dict(method='zip',
                        archive_name='results_frugal.csv')
results_frugal.to_csv('results_frugal.zip', index=False,
          compression=compression_opts)