# Reading in data (pickle) and perform data analysis

import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Specify the directory in which you stored the results
entries = Path('/Users/Jacqueline/Documents/Research_Internship/Implementation/Simulations/Results/Simulations/')

# Initialize pandas dataframe to store results
results = pd.DataFrame(
    columns=['Intention Speaker', 'Inferred Referent Listener', 'Number of Turns', 'Order of Reasoning Speaker',
             'Order of Reasoning Listener', 'Communicative Success', 'Reached Threshold Order',
             'Reached Threshold Interaction', 'Dialogue History', 'Ambiguity Level', 'Number of Signals',
             'Number of Referents', 'Entropy Threshold'])

# Read in all files of the directory in which you stored the results and concatenate them in order to visualise
for entry in entries.iterdir():
    if str(entry).startswith('/Users/Jacqueline/Documents/Research_Internship/Implementation/Simulations/Results/'
                             'Simulations/amb'):
        result = pd.read_pickle(entry)
        results = pd.concat([results, result])

# Make sure that the values are integers in order to take the mean
results["Communicative Success"] = results["Communicative Success"].astype(int)
results["Number of Turns"] = results["Number of Turns"].astype(int)
results["Ambiguity Level"] = results["Ambiguity Level"].astype(float)
results["Number of Signals"] = results["Number of Signals"].astype(int)

# Take the results of the interactional and pragmatic agents seperately
results_interactional = results[results["Order of Reasoning Listener"] == 0]
results_pragmatic = results[results["Order of Reasoning Listener"] != 0]

# Replace the entropy threshold for pragmatic agents with order 2 with NaN as it's not applicable
results.loc[results["Order of Reasoning Listener"] == 2, "Entropy Threshold"] = np.nan
print(results[results["Order of Reasoning Listener"] == 2])

# Plot the results
#g = sns.FacetGrid(results, col=2)
# fig, ax = plt.subplots(1,2)
# sns.barplot(x="Entropy Threshold", y="Number of Turns" , data=results, hue="Order of Reasoning Listener", ax=ax[0])
# sns.barplot(x="Entropy Threshold", y="Communicative Success", data=results, hue="Order of Reasoning Listener", ax=ax[1])


# sns.relplot(x="Ambiguity Level", y="Communicative Success", hue="Number of Signals",
#              size="Number of Turns", facet_kws=dict(sharex=False), kind="line", legend="brief",
#              data=results)

#sns.catplot(x="Ambiguity Level", y="Communicative Success", hue="Number of Signals", col="Entropy Threshold",
#            kind="bar", data=results_pragmatic)

plt.show()
