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

# Make sure that the values are integers/floats in order to take the mean
results["Communicative Success"] = results["Communicative Success"].astype(int)
results["Number of Turns"] = results["Number of Turns"].astype(int)
results["Ambiguity Level"] = results["Ambiguity Level"].astype(float)
results["Number of Signals"] = results["Number of Signals"].astype(int)

# Change the number of turns to current definition (listener interpretation is not a turn)
results["Number of Turns"] = results["Number of Turns"] - 1

# Replace the entropy threshold for pragmatic agents with order 2 with NaN as it's not applicable
results.loc[results["Order of Reasoning Listener"] == 2, "Entropy Threshold"] = np.nan

# Take the results of the interactional and pragmatic agents separately
results_interactional = results[results["Order of Reasoning Listener"] == 0]
results_pragmatic = results[results["Order of Reasoning Listener"] != 0]

# Different pragmatic agents
results_1 = results_pragmatic[(results_pragmatic["Order of Reasoning Listener"] == 1) & (results_pragmatic["Reached Threshold Order"]==False)]
results_2 = results_pragmatic[results_pragmatic["Order of Reasoning Listener"] == 2]
results_frugal = results_pragmatic[(results_pragmatic["Order of Reasoning Listener"] == 1) & (results_pragmatic["Reached Threshold Order"]==True)]

# Pickle the results to be stored and used later for plotting
filename = '../Results/results.p'
outfile = open(filename, 'wb')
pickle.dump(results, outfile)
outfile.close()

filename = '../Results/results_interactional.p'
outfile = open(filename, 'wb')
pickle.dump(results_interactional, outfile)
outfile.close()

filename = '../Results/results_pragmatic.p'
outfile = open(filename, 'wb')
pickle.dump(results_pragmatic, outfile)
outfile.close()

filename = '../Results/results_1.p'
outfile = open(filename, 'wb')
pickle.dump(results_1, outfile)
outfile.close()

filename = '../Results/results_2.p'
outfile = open(filename, 'wb')
pickle.dump(results_2, outfile)
outfile.close()

filename = '../Results/results_frugal.p'
outfile = open(filename, 'wb')
pickle.dump(results_frugal, outfile)
outfile.close()