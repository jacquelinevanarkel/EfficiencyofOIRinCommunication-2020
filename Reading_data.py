# Necessary imports
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------- Reading in data & preprocessing before plotting -----------------------------------
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

# Make sure that the values are integers/floats in order to take the mean
results["Communicative Success"] = results["Communicative Success"].astype(int)
results["Number of Turns"] = results["Number of Turns"].astype(int)
results["Ambiguity Level"] = results["Ambiguity Level"].astype(float)
results["Number of Signals"] = results["Number of Signals"].astype(int)

# Take the results of the interactional and pragmatic agents separately
results_interactional = results[results["Order of Reasoning Listener"] == 0]
results_pragmatic = results[results["Order of Reasoning Listener"] != 0]

# Replace the entropy threshold for pragmatic agents with order 2 with NaN as it's not applicable
results.loc[results["Order of Reasoning Listener"] == 2, "Entropy Threshold"] = np.nan

# ------------------------------------------------- Plot the results ---------------------------------------------------

# Plot the communicative success per lexicon size with lines for the chance levels & relative CS
CS_means = results.groupby("Number of Referents")["Communicative Success"].mean()

plt.subplot(1, 2, 1)
Bars = plt.bar(np.arange(3), CS_means, align='center')
_ = plt.xticks(np.arange(3), ["4", "10", "20"])
plt.ylim(0, 1)
plt.title("CS for Different Lexicon Sizes")
plt.ylabel("Communicative Success")
plt.xlabel("Number of Referents Lexicon")

chance_levels = (0.25, 0.1, 0.05)
x_start = np.array([plt.getp(item, 'x') for item in Bars])
x_end = x_start+[plt.getp(item, 'width') for item in Bars]

plt.hlines(chance_levels, x_start, x_end)

plt.subplot(1, 2, 2)
Bars = plt.bar(np.arange(3), CS_means-chance_levels, align='center')
_ = plt.xticks(np.arange(3), ["4", "10", "20"])
plt.ylim(0, 1)
plt.title("CS for Different Lexicon Sizes - Chance Levels per Lexicon Size")
plt.ylabel("Relative Communicative Success")
plt.xlabel("Number of Referents Lexicon")

# Used for other plots

#fig, ax = plt.subplots(1,2)
#sns.barplot(x="Entropy Threshold", y="Number of Turns" , data=results, hue="Order of Reasoning Listener", ax=ax[0])
#sns.barplot(x="Entropy Threshold", y="Communicative Success", data=results, hue="Order of Reasoning Listener", ax=ax[1)

#sns.catplot(x="Ambiguity Level", y="Communicative Success", hue="Number of Signals", col="Entropy Threshold",
#            kind="bar", data=results_pragmatic)

plt.show()
