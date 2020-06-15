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

# Replace the entropy threshold for pragmatic agents with order 2 with NaN as it's not applicable
results.loc[results["Order of Reasoning Listener"] == 2, "Entropy Threshold"] = np.nan

# Take the results of the interactional and pragmatic agents separately
results_interactional = results[results["Order of Reasoning Listener"] == 0]
results_pragmatic = results[results["Order of Reasoning Listener"] != 0]

# Pickle the results to be stored and used later for plotting
filename = 'results.p'
outfile = open(filename, 'wb')
pickle.dump(results, outfile)
outfile.close()

filename = 'results_interactional.p'
outfile = open(filename, 'wb')
pickle.dump(results_interactional, outfile)
outfile.close()

filename = 'results_pragmatic.p'
outfile = open(filename, 'wb')
pickle.dump(results_pragmatic, outfile)
outfile.close()