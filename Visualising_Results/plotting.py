# Necessary imports
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import matplotlib as mpl

# Read in data
results = pd.read_pickle("../Results/results.p")
results_interactional = pd.read_pickle("../Results/results_interactional.p")
results_pragmatic = pd.read_pickle("../Results/results_pragmatic.p")
results_1 = pd.read_pickle("../Results/results_1.p")
results_2 = pd.read_pickle("../Results/results_2.p")
results_frugal = pd.read_pickle("../Results/results_frugal.p")

# The first part of the plotting is done using matplotlib. The second part of the plotting is done with the help of the
# seaborn package in combination with matplotlib.

# ------------------------------------------------- Plot the results ---------------------------------------------------
# Plot the communicative success per lexicon size with lines for the chance levels & relative CS

# Calculate the means of the communicative success for the interactional and pragmatic agents
CS_means_interactional = results_interactional.groupby(["Number of Referents", "Ambiguity Level"])["Communicative Success"].mean()
CS_means_pragmatic = results_pragmatic.groupby(["Number of Referents","Ambiguity Level"])["Communicative Success"].mean()

# Calculate SDs
SDs = []
size = results_interactional.groupby(["Number of Referents", "Ambiguity Level"]).size()
for index, value in CS_means_interactional.items():
    P = value
    N = size[index]
    SD = ((P*(1-P))/N)**0.5
    SDs.append(SD)

plt.subplot(1, 2, 1)

# set width of bar
barWidth = 0.25

# set height of bar
bars1 = CS_means_interactional[:3]
bars2 = CS_means_interactional[3:6]
bars3 = CS_means_interactional[6:9]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=[x * 2 for x in SDs[:3]])
plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=[x * 2 for x in SDs[3:6]])
plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=[x * 2 for x in SDs[6:9]])

# Add xticks on the middle of the group bars
plt.xlabel('Ambiguity Level', fontweight='bold')
plt.ylabel('Communicative Success', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['0.2','0.5','0.8'])

# Add titles
plt.suptitle("Communicative Success for Different Ambiguity levels and Lexicon Sizes")
plt.title("Interactional Agents")

# Add chance levels
chance_levels = (0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05)
x_start = [r1[0]-(barWidth/2), r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(barWidth/2), r2[1]-(barWidth/2),
           r3[1]-(barWidth/2), r1[2]-(barWidth/2), r2[2]-(barWidth/2), r3[2]-(barWidth/2)]
x_end = [r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(1.5*barWidth), r2[1]-(barWidth/2), r3[1]-(barWidth/2),
         r1[2]-(1.5*barWidth), r2[2]-(barWidth/2), r3[2]-(barWidth/2), r3[2]+barWidth-(barWidth/2)]
plt.hlines(chance_levels, x_start, x_end)

# Start second plot
plt.subplot(1, 2, 2)

# Calculate SDs
SDs = []
size = results_pragmatic.groupby(["Number of Referents", "Ambiguity Level"]).size()
for index, value in CS_means_pragmatic.items():
    P = value
    N = size[index]
    SD = ((P*(1-P))/N)**0.5
    SDs.append(SD)

# set width of bar
barWidth = 0.25

# set height of bar
bars1 = CS_means_pragmatic[:3]
bars2 = CS_means_pragmatic[3:6]
bars3 = CS_means_pragmatic[6:9]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=[x * 2 for x in SDs[:3]])
plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=[x * 2 for x in SDs[3:6]])
plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=[x * 2 for x in SDs[6:9]])

# Add xticks on the middle of the group bars
plt.xlabel('Ambiguity Level', fontweight='bold')
plt.ylabel('Communicative Success', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['0.2','0.5','0.8'])

plt.title("Pragmatic Agents")

# Add chance levels
chance_levels = (0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05)
x_start = [r1[0]-(barWidth/2), r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(barWidth/2), r2[1]-(barWidth/2),
           r3[1]-(barWidth/2), r1[2]-(barWidth/2), r2[2]-(barWidth/2), r3[2]-(barWidth/2)]
x_end = [r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(1.5*barWidth), r2[1]-(barWidth/2), r3[1]-(barWidth/2),
         r1[2]-(1.5*barWidth), r2[2]-(barWidth/2), r3[2]-(barWidth/2), r3[2]+barWidth-(barWidth/2)]
plt.hlines(chance_levels, x_start, x_end)

# Create legend & Show graphic
plt.legend(title="Lexicon Size")
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Number of turns interactional agents

# Calculate the mean and standard deviation of the number of turns for the interactional agents
turns_means_interactional = results_interactional.groupby(["Number of Referents", "Ambiguity Level"])["Number of Turns"].mean()
turns_sd_interactional = results_interactional.groupby(["Number of Referents", "Ambiguity Level"])["Number of Turns"].std()

# set width of bar
barWidth = 0.25

# set height of bar
bars1 = turns_means_interactional[:3]
bars2 = turns_means_interactional[3:6]
bars3 = turns_means_interactional[6:9]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=2*turns_sd_interactional[:3])
plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=2*turns_sd_interactional[3:6])
plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=2*turns_sd_interactional[6:9])

# Add xticks on the middle of the group bars
plt.xlabel('Ambiguity Level', fontweight='bold')
plt.ylabel('Number of Turns', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['0.2','0.5','0.8'])

# Add titles
plt.title("Number of Turns for Different Ambiguity levels and Lexicon Sizes")
plt.legend(title="Lexicon Size")

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Plot of the effect of order of reasoning and lexicon size on CS

# Take the means of the CS grouped by the lexicon size and the order of reasoning
CS_means = results.groupby(["Number of Referents", "Order of Reasoning Listener"])["Communicative Success"].mean()

# Calculate SDs
SDs = []
size = results.groupby(["Number of Referents", "Order of Reasoning Listener"])["Communicative Success"].size()
for index, value in CS_means.items():
    P = value
    N = size[index]
    SD = ((P*(1-P))/N)**0.5
    SDs.append(SD)

# set width of bar
barWidth = 0.25

# set height of bar
bars1 = CS_means[:3]
bars2 = CS_means[3:6]
bars3 = CS_means[6:9]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=[x * 2 for x in SDs[:3]])
plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=[x * 2 for x in SDs[3:6]])
plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=[x * 2 for x in SDs[6:9]])

# Add xticks on the middle of the group bars
plt.xlabel('Order of Pragmatic Reasoning', fontweight='bold')
plt.ylabel('Communicative Success', fontweight='bold')
plt.ylim(0,1)
plt.xticks([r + barWidth for r in range(len(bars1))], ['0','1','2'])

# Add chance levels
chance_levels = (0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05)
x_start = [r1[0]-(barWidth/2), r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(barWidth/2), r2[1]-(barWidth/2),
           r3[1]-(barWidth/2), r1[2]-(barWidth/2), r2[2]-(barWidth/2), r3[2]-(barWidth/2)]
x_end = [r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(1.5*barWidth), r2[1]-(barWidth/2), r3[1]-(barWidth/2),
         r1[2]-(1.5*barWidth), r2[2]-(barWidth/2), r3[2]-(barWidth/2), r3[2]+barWidth-(barWidth/2)]
plt.hlines(chance_levels, x_start, x_end)

# Add titles
plt.title("Communicative Success for Different Orders of Pragmatic Reasoning and Lexicon Sizes")
plt.legend(title="Lexicon Size")
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Plots of the CS for the entropy threshold and lexicon size

# Get the means and standard deviations for the CS and turns grouped by the lexicon size and entorpy threshold
CS_means = results_interactional.groupby(["Number of Referents", "Entropy Threshold"])["Communicative Success"].mean()
CS_means_pragmatic = results_pragmatic.groupby(["Number of Referents", "Entropy Threshold"])["Communicative Success"].mean()

plt.subplot(1, 2, 1)

# Calculate SDs for the CS
SDs = []
size = results_interactional.groupby(["Number of Referents", "Entropy Threshold"])["Communicative Success"].size()
for index, value in CS_means.items():
    P = value
    N = size[index]
    SD = ((P*(1-P))/N)**0.5
    SDs.append(SD)

# set width of bar
barWidth = 0.25

# set height of bar
bars1 = CS_means[:3]
bars2 = CS_means[3:6]
bars3 = CS_means[6:9]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=[x * 2 for x in SDs[:3]])
plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=[x * 2 for x in SDs[3:6]])
plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=[x * 2 for x in SDs[6:9]])

# Add xticks on the middle of the group bars
plt.xlabel('Entropy Threshold', fontweight='bold')
plt.ylabel('Communicative Success', fontweight='bold')
plt.ylim(0,1)
plt.xticks([r + barWidth for r in range(len(bars1))], ['0.8','1.0','1.5'])

# Add chance levels
chance_levels = (0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05)
x_start = [r1[0]-(barWidth/2), r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(barWidth/2), r2[1]-(barWidth/2),
           r3[1]-(barWidth/2), r1[2]-(barWidth/2), r2[2]-(barWidth/2), r3[2]-(barWidth/2)]
x_end = [r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(1.5*barWidth), r2[1]-(barWidth/2), r3[1]-(barWidth/2),
         r1[2]-(1.5*barWidth), r2[2]-(barWidth/2), r3[2]-(barWidth/2), r3[2]+barWidth-(barWidth/2)]
plt.hlines(chance_levels, x_start, x_end)

# Add titles
plt.suptitle("Communicative Success for Different Entropy Thresholds and Lexicon Sizes")
plt.title("Interactional Agents")
plt.legend(title="Lexicon Size")

plt.subplot(1, 2, 2)

# Calculate SDs
SDs = []
size = results_pragmatic.groupby(["Number of Referents", "Entropy Threshold"])["Communicative Success"].size()
for index, value in CS_means_pragmatic.items():
    P = value
    N = size[index]
    SD = ((P*(1-P))/N)**0.5
    SDs.append(SD)

# set width of bar
barWidth = 0.25

# set height of bar
bars1 = CS_means_pragmatic[:3]
bars2 = CS_means_pragmatic[3:6]
bars3 = CS_means_pragmatic[6:9]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=[x * 2 for x in SDs[:3]])
plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=[x * 2 for x in SDs[3:6]])
plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=[x * 2 for x in SDs[6:9]])

# Add xticks on the middle of the group bars
plt.xlabel('Entropy Threshold', fontweight='bold')
plt.ylabel('Communicative Success', fontweight='bold')
plt.ylim(0,1)
plt.xticks([r + barWidth for r in range(len(bars1))], ['0.8','1.0','1.5'])

# Add chance levels
chance_levels = (0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05)
x_start = [r1[0]-(barWidth/2), r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(barWidth/2), r2[1]-(barWidth/2),
           r3[1]-(barWidth/2), r1[2]-(barWidth/2), r2[2]-(barWidth/2), r3[2]-(barWidth/2)]
x_end = [r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(1.5*barWidth), r2[1]-(barWidth/2), r3[1]-(barWidth/2),
         r1[2]-(1.5*barWidth), r2[2]-(barWidth/2), r3[2]-(barWidth/2), r3[2]+barWidth-(barWidth/2)]
plt.hlines(chance_levels, x_start, x_end)

# Add titles
plt.title("Pragmatic Agents")
plt.legend(title="Lexicon Size")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Plot of the mean number of turns for the interactional agents grouped by the lexicon size and entropy threshold

# Get the means and standard deviations
turns_means = results.groupby(["Number of Referents", "Entropy Threshold"])["Number of Turns"].mean()
turns_sd = results.groupby(["Number of Referents", "Entropy Threshold"])["Number of Turns"].std()

# set width of bar
barWidth = 0.25

# set height of bar
bars1 = turns_means[:3]
bars2 = turns_means[3:6]
bars3 = turns_means[6:9]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=2*turns_sd[:3])
plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=2*turns_sd[3:6])
plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=2*turns_sd[6:9])

# Add xticks on the middle of the group bars
plt.xlabel('Entropy Threshold', fontweight='bold')
plt.ylabel('Number of Turns', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['0.8','1.0','1.5'])

plt.title("Number of Turns for Different Entropy Thresholds and Lexicon Sizes")
plt.legend(title="Lexicon Size")
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Order of reasoning on CS - where listener ended

# Get means of CS grouped by the lexicon size and order of reasoning
results["Order of Reasoning"] = np.where(results["Reached Threshold Order"]==True, 2, results["Order of Reasoning Listener"])
CS_means = results.groupby(["Number of Referents", "Order of Reasoning"])["Communicative Success"].mean()

# Calculate SDs
SDs = []
size = results.groupby(["Number of Referents", "Order of Reasoning"])["Communicative Success"].size()
for index, value in CS_means.items():
    P = value
    N = size[index]
    SD = ((P*(1-P))/N)**0.5
    SDs.append(SD)

plt.subplot(1, 2, 1)

# set width of bar
barWidth = 0.25

# set height of bar
bars1 = CS_means[:3]
bars2 = CS_means[3:6]
bars3 = CS_means[6:9]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=[x * 2 for x in SDs[:3]])
plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=[x * 2 for x in SDs[3:6]])
plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=[x * 2 for x in SDs[6:9]])

# Add xticks on the middle of the group bars
plt.xlabel('Order of Pragmatic Reasoning', fontweight='bold')
plt.ylabel('Communicative Success', fontweight='bold')
plt.ylim(0,1)
plt.xticks([r + barWidth for r in range(len(bars1))], ['0','1','2'])

# Add chance levels
chance_levels = (0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05)
x_start = [r1[0]-(barWidth/2), r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(barWidth/2), r2[1]-(barWidth/2),
           r3[1]-(barWidth/2), r1[2]-(barWidth/2), r2[2]-(barWidth/2), r3[2]-(barWidth/2)]
x_end = [r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(1.5*barWidth), r2[1]-(barWidth/2), r3[1]-(barWidth/2),
         r1[2]-(1.5*barWidth), r2[2]-(barWidth/2), r3[2]-(barWidth/2), r3[2]+barWidth-(barWidth/2)]
plt.hlines(chance_levels, x_start, x_end)

# Add titles
plt.title("Communicative Success \n for Different Orders of Pragmatic Reasoning and Lexicon Sizes")
plt.legend(title="Lexicon Size")

# Start second plot
plt.subplot(1, 2, 2)

# set width of bar
barWidth = 0.25

# set height of bar
bars1 = size[:3]
bars2 = size[3:6]
bars3 = size[6:9]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4')
plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10')
plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20')

# Add xticks on the middle of the group bars and set labels
plt.xlabel('Order of Pragmatic Reasoning', fontweight='bold')
plt.ylabel('Amount of Interactions', fontweight='bold')
plt.ylim(0,480000)
plt.xticks([r + barWidth for r in range(len(bars1))], ['0','1','2'])

# Add titles
plt.title("Amount of Interactions \n for Different Orders of Pragmatic Reasoning and Lexicon Sizes")
plt.legend(title="Lexicon Size")
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# From here one, seaborn is used to make plots
# ----------------------------------------------------------------------------------------------------------------------

# Plot of the mean entropy difference over time in an interaction for the interactional agents

# Get the entropy difference and add it to a list
data = []
for entropy in results_interactional["Entropy"]:
    if len(entropy) == 30:
        difference = np.diff(np.array(entropy))
    else:
        new = list(itertools.chain(np.array(entropy), [np.NAN] * (30 - len(entropy))))
        difference = np.diff(np.array(new))
    data.append(difference)

# Change the sign of the entropy difference
mean_ent_diff = np.negative(np.nanmean(data, axis=0))

# Plot the entropy difference
sns.lineplot(data=mean_ent_diff)
plt.title("Mean Entropy Difference Over Interactions")
plt.xticks(np.arange(29), np.arange(2, 31))
plt.ylabel("Entropy Difference over Turns (n-1)")
plt.xlabel("Turn")
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Plot of the mean entropy difference over time in an interaction for the interactional agents for different lexicon sizes

results_interactional["Entropy Difference"] = np.where(len(results_interactional["Entropy"])==30, np.diff(np.array(results_interactional["Entropy"])),
                                                      np.diff(np.array(list(itertools.chain(np.array(results_interactional["Entropy"]), [np.NAN] * (30 - len(results_interactional["Entropy"])))))))

# Create different subsets of the data per lexicon size
lex0 = results_interactional[results_interactional["Number of Referents"] == 4]
lex1 = results_interactional[results_interactional["Number of Referents"] == 10]
lex2 = results_interactional[results_interactional["Number of Referents"] == 20]

# Initialise empty lists for the data of the different lexicon sizes
data_interactional_0 = []
data_interactional_1 = []
data_interactional_2 = []

# Calculate the entropy difference
for entropy in lex0["Entropy"]:
    if len(entropy) == 6:
        difference = np.diff(np.array(entropy))
    else:
        new = list(itertools.chain(np.array(entropy), [np.NAN] * (6 - len(entropy))))
        difference = np.diff(np.array(new))
    data_interactional_0.append(difference)

for entropy in lex1["Entropy"]:
    if len(entropy) == 15:
        difference = np.diff(np.array(entropy))
    else:
        new = list(itertools.chain(np.array(entropy), [np.NAN] * (15 - len(entropy))))
        difference = np.diff(np.array(new))
    data_interactional_1.append(difference)

for entropy in lex2["Entropy"]:
    if len(entropy) == 30:
        difference = np.diff(np.array(entropy))
    else:
        new = list(itertools.chain(np.array(entropy), [np.NAN] * (30 - len(entropy))))
        difference = np.diff(np.array(new))
    data_interactional_2.append(difference)

# Change the sign of the entropy difference per lexicon size
mean_ent_diff_interactional_lex1 = np.negative(np.nanmean(data_interactional_0, axis=0))
mean_ent_diff_interactional_lex2 = np.negative(np.nanmean(data_interactional_1, axis=0))
mean_ent_diff_interactional_lex3 = np.negative(np.nanmean(data_interactional_2, axis=0))

# Define the colors to be used for the plot
colors = ['#fdcc8a', '#fc8d59', '#d7301f']

# Create plot
sns.set_style("whitegrid")
sns.lineplot(data=mean_ent_diff_interactional_lex1, color=colors[0])
plt.plot(mean_ent_diff_interactional_lex2, color=colors[1])
plt.plot(mean_ent_diff_interactional_lex3, color=colors[2])

# Change ticks and labels and legend
plt.xticks(np.arange(29, step=2), np.arange(2, 31, step=2), fontsize=18)
plt.ylabel("Entropy Reduction over Turns", fontsize=18, fontweight='bold')
plt.xlabel("Turn", fontsize=18, fontweight='bold')
plt.yticks(fontsize=18)
handles, labels = plt.gca().get_legend_handles_labels()
legend = plt.legend(handles, labels=["6x4", "15x10", "30x20"], title="Lexicon Size", fontsize=16)
plt.setp(legend.get_title(), fontsize=16, fontweight="bold")

plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# CS plots for the interactional and pragmatic agents by the lexicon size

fig, ax = plt.subplots(1,2)

# The lines for the chance levels
ax[0].hlines(y=[0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05],xmin=[-0.4, -0.13, 0.13, 0.60, 0.87, 1.13, 1.60, 1.87, 2.13],
           xmax=[-0.13, 0.13, 0.40, 0.87, 1.13, 1.40, 1.87, 2.13, 2.40])

# Create the plots and set labels and titles
sns.barplot(x="Ambiguity Level", y="Communicative Success", hue="Number of Referents", data=results_interactional, ax=ax[0])
ax[0].set_ylabel("Mean Communicative Success")
ax[0].set_title("Interactional Agents")
ax[0].set_ylim(0,1)

# The lines for the chance levels
ax[1].hlines(y=[0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05],xmin=[-0.4, -0.13, 0.13, 0.60, 0.87, 1.13, 1.60, 1.87, 2.13],
           xmax=[-0.13, 0.13, 0.40, 0.87, 1.13, 1.40, 1.87, 2.13, 2.40])

# Create the plots and set labels and titles
sns.barplot(x="Ambiguity Level", y="Communicative Success", hue="Number of Referents", data=results_pragmatic, ax=ax[1])
ax[1].set_title("Pragmatic Agents")
ax[1].set_ylabel("Mean Communicative Success")
ax[1].set_ylim(0,1)

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Plot of the number of turns for the lexicon size and ambiguity level

# Create plots
sns.violinplot(x="Ambiguity Level", y="Number of Turns", hue="Number of Referents", data=results_interactional)
sns.stripplot(x="Ambiguity Level", y="Number of Turns", hue="Number of Referents", data=results_interactional, dodge=True)
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Plot of the number of turns of the interactional agents for the different lexicon sizes and for the selected
# ambiguity level and entropy threshold

# Select the data based on an ambiguity level of 0.5 and entropy threshold of 1.0
results_selected = results_interactional[(results_interactional["Ambiguity Level"] == 0.5) & (results_interactional["Entropy Threshold"] != 0.8)]
results_selected = results_selected[results_selected["Entropy Threshold"] != 1.5]

# Specify the colours to use
colors = ['#fdcc8a', '#fc8d59', '#d7301f']
sns.set_palette(sns.color_palette(colors))
sns.set_style("whitegrid")

# Plot
sns.violinplot(x="Number of Referents", y="Number of Turns", data=results_selected, linewidth=4)
sns.stripplot(x="Number of Referents", y="Number of Turns", data=results_selected, dodge=True, alpha=0.2)

# Set ticks and labels
plt.yticks(fontsize=18)
locs, labels = plt.xticks()
plt.xticks(locs, (["6x4", "15x10", "30x20"]), fontsize=18)
plt.ylabel("Number of Turns", fontsize=18, fontweight="bold")
plt.xlabel("Lexicon Size", fontsize=18, fontweight="bold")

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Plot of the communicative success for the different lexicon sizes and for the selected ambiguity level
# and entropy threshold

# Select the results based on an ambiguity level of 0.5 and entropy threshold of 1.0
results_selected = results[(results["Ambiguity Level"] == 0.5) & (results["Entropy Threshold"] != 0.8)]
results_selected = results_selected[results_selected["Entropy Threshold"] != 1.5]

# Specify the colours to use for plotting
colors = ['#fdcc8a', '#fc8d59', '#d7301f']
sns.set_palette(sns.color_palette(colors))

# Plot
sns.barplot(x="Order of Reasoning Listener", y="Communicative Success", hue="Number of Referents", data=results_selected)

# Set ticks and labels and titles
plt.ylim(0,1)
plt.yticks(fontsize=22)
handles, labels = plt.gca().get_legend_handles_labels()
legend = plt.legend(handles, (["6x4", "15x10", "30x20"]), title="Lexicon Size", fontsize=22)
plt.setp(legend.get_title(), fontsize=22, fontweight="bold")
locs, labels = plt.xticks()
plt.xticks(locs, labels=["Interactional", "Frugally Pragmatic", "Fully Pragmatic"], fontsize=22)
plt.ylabel("Mean Communicative Success", fontsize=22, fontweight="bold")
plt.xlabel("Agent Type", fontsize=22, fontweight="bold")

# Add chance levels
plt.hlines(y=[0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05],
           xmin=[-0.4, -0.13, 0.13, 0.60, 0.87, 1.13, 1.60, 1.87, 2.13],
           xmax=[-0.13, 0.13, 0.40, 0.87, 1.13, 1.40, 1.87, 2.13, 2.40])

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Plots of the CS compared between interactional and pragmatic agents, per lexicon size and entropy threshold

fig, ax = plt.subplots(1,2)

# Add chance levels
ax[0].hlines(y=[0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05],xmin=[-0.4, -0.13, 0.13, 0.60, 0.87, 1.13, 1.60, 1.87, 2.13],
           xmax=[-0.13, 0.13, 0.40, 0.87, 1.13, 1.40, 1.87, 2.13, 2.40])

# Plot
sns.barplot(x="Entropy Threshold", y="Communicative Success", hue="Number of Referents", data=results_interactional, ax=ax[0])

# Set labels and titles
ax[0].set_ylabel("Mean Communicative Success")
ax[0].set_ylim(0,1)
ax[0].set_title("Interactional Agents")

# Add chance levels for second plot
ax[1].hlines(y=[0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05],xmin=[-0.4, -0.13, 0.13, 0.60, 0.87, 1.13, 1.60, 1.87, 2.13],
           xmax=[-0.13, 0.13, 0.40, 0.87, 1.13, 1.40, 1.87, 2.13, 2.40])

# Plot second plot
sns.barplot(x="Entropy Threshold", y="Communicative Success", hue="Number of Referents", data=results_pragmatic, ax=ax[1])

# Set titles and labels
ax[1].set_title("Pragmatic Agents")
ax[1].set_ylabel("Mean Communicative Success")
ax[1].set_ylim(0,1)


plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Number of turns of the interactional agents over different lexicon sizes and entropy thresholds

# Plot
sns.violinplot(x="Entropy Threshold", y="Number of Turns", hue="Number of Referents", data=results_interactional)

# Specify legend
plt.legend(loc='upper left', title="Number of Referents")

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Summarising the results by creating a heatmap of the communicative success for the different lexicon sizes, agent types,
# entropy thresholds and ambiguity levels

# Specify the different groups of data
lex1 = results_interactional[results_interactional["Number of Referents"] == 4]
lex2 = results_interactional[results_interactional["Number of Referents"] == 10]
lex3 = results_interactional[results_interactional["Number of Referents"] == 20]
data1 = pd.pivot_table(lex1, values='Communicative Success', index=['Entropy Threshold'], columns='Ambiguity Level')
data2 = pd.pivot_table(lex2, values='Communicative Success', index=['Entropy Threshold'], columns='Ambiguity Level')
data3 = pd.pivot_table(lex3, values='Communicative Success', index=['Entropy Threshold'], columns='Ambiguity Level')

lex4 = results_1[results_1["Number of Referents"] == 4]
lex5 = results_1[results_1["Number of Referents"] == 10]
lex6 = results_1[results_1["Number of Referents"] == 20]
data4 = pd.pivot_table(lex4, values='Communicative Success', index=['Entropy Threshold'], columns='Ambiguity Level')
data5 = pd.pivot_table(lex5, values='Communicative Success', index=['Entropy Threshold'], columns='Ambiguity Level')
data6 = pd.pivot_table(lex6, values='Communicative Success', index=['Entropy Threshold'], columns='Ambiguity Level')

lex7 = results_frugal[results_frugal["Number of Referents"] == 4]
lex8 = results_frugal[results_frugal["Number of Referents"] == 10]
lex9 = results_frugal[results_frugal["Number of Referents"] == 20]
data7 = pd.pivot_table(lex7, values='Communicative Success', index=['Entropy Threshold'], columns='Ambiguity Level')
data8 = pd.pivot_table(lex8, values='Communicative Success', index=['Entropy Threshold'], columns='Ambiguity Level')
data9 = pd.pivot_table(lex9, values='Communicative Success', index=['Entropy Threshold'], columns='Ambiguity Level')

lex10 = results_2[results_2["Number of Referents"] == 4]
lex11 = results_2[results_2["Number of Referents"] == 10]
lex12 = results_2[results_2["Number of Referents"] == 20]
data10 = pd.pivot_table(lex10, values='Communicative Success', columns='Ambiguity Level')
data11 = pd.pivot_table(lex11, values='Communicative Success', columns='Ambiguity Level')
data12 = pd.pivot_table(lex12, values='Communicative Success', columns='Ambiguity Level')

# Start plotting
fig, ax = plt.subplots(4,3)
sns.heatmap(data1, ax=ax[0,0], vmin=0, vmax=1, cmap="YlGnBu", cbar=False, annot=True)
sns.heatmap(data2, ax=ax[0,1], vmin=0, vmax=1, cmap="YlGnBu", cbar=False, annot=True)
sns.heatmap(data3, ax=ax[0,2], vmin=0, vmax=1, cmap="YlGnBu", annot=True)

# Add titles
ax[0,0].set_title("Lexicon: 6x4")
ax[0,1].set_title("Interactional Agents \n Lexicon: 15x10")
ax[0,2].set_title("Lexicon: 30x20")
ax[1,0].set_title("Lexicon: 6x4")
ax[1,1].set_title(" \n Frugally Pragmatic Agents 1 \n Lexicon: 15x10")
ax[1,2].set_title("Lexicon: 30x20")
ax[2,0].set_title("Lexicon: 6x4")
ax[2,1].set_title(" \n Frugally Pragmatic Agents 2 \n Lexicon: 15x10")
ax[2,2].set_title("Lexicon: 30x20")
ax[3,0].set_title("Lexicon: 6x4")
ax[3,1].set_title(" \n Fully Pragmatic Agents \n Lexicon: 15x10")
ax[3,2].set_title("Lexicon: 30x20")

# Plot
sns.heatmap(data4, ax=ax[1,0], vmin=0, vmax=1, cmap="YlGnBu", cbar=False, annot=True)
sns.heatmap(data5, ax=ax[1,1], vmin=0, vmax=1, cmap="YlGnBu", cbar=False,  annot=True)
sns.heatmap(data6, ax=ax[1,2], vmin=0, vmax=1, cmap="YlGnBu",  annot=True)

sns.heatmap(data7, ax=ax[2,0], vmin=0, vmax=1, cmap="YlGnBu", cbar=False, annot=True)
sns.heatmap(data8, ax=ax[2,1], vmin=0, vmax=1, cmap="YlGnBu", cbar=False, annot=True)
sns.heatmap(data9, ax=ax[2,2], vmin=0, vmax=1, cmap="YlGnBu", annot=True)

sns.heatmap(data10, ax=ax[3,0], vmin=0, vmax=1, cmap="YlGnBu", cbar=False, annot=True, yticklabels=False)
sns.heatmap(data11, ax=ax[3,1], vmin=0, vmax=1, cmap="YlGnBu", cbar=False, annot=True, yticklabels=False)
sns.heatmap(data12, ax=ax[3,2], vmin=0, vmax=1, cmap="YlGnBu", annot=True, yticklabels=False)

# Remove some labels
ax[0,1].set_ylabel('')
ax[0,1].set_xlabel('')
ax[0,2].set_ylabel('')
ax[0,2].set_xlabel('')
ax[1,1].set_ylabel('')
ax[0,0].set_xlabel('')
ax[1,2].set_ylabel('')
ax[2,1].set(ylabel='', xlabel='')
ax[2,2].set(ylabel='', xlabel='')
ax[2,0].set(xlabel='')
ax[1,2].set_xlabel('')
ax[1,0].set_xlabel('')
ax[1,1].set_xlabel('')
ax[3,0].set_ylabel('No Entropy Threshold')

fig.subplots_adjust(hspace=0.55)

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Summarise results by creating a heatmap of the number of turns for the interactional agents for the different ambiguity
# levels, entropy thresholds and lexicon sizes

# Select groups of data
lex1 = results_interactional[results_interactional["Number of Referents"] == 4]
lex2 = results_interactional[results_interactional["Number of Referents"] == 10]
lex3 = results_interactional[results_interactional["Number of Referents"] == 20]
data1 = pd.pivot_table(lex1, values='Number of Turns', index=['Entropy Threshold'], columns='Ambiguity Level')
data2 = pd.pivot_table(lex2, values='Number of Turns', index=['Entropy Threshold'], columns='Ambiguity Level')
data3 = pd.pivot_table(lex3, values='Number of Turns', index=['Entropy Threshold'], columns='Ambiguity Level')

# Plot
fig, ax = plt.subplots(1,3)
sns.heatmap(data1, ax=ax[0], vmin=0, vmax=26, cmap="YlGnBu", cbar=False, annot=True)
sns.heatmap(data2, ax=ax[1], vmin=0, vmax=26, cmap="YlGnBu", cbar=False, annot=True)
sns.heatmap(data3, ax=ax[2], vmin=0, vmax=26, cmap="YlGnBu", annot=True)

# Set titles and labels
ax[0].set_title("Lexicon: 6x4")
ax[1].set_title("Lexicon: 15x10")
ax[2].set_title("Lexicon: 30x20")
ax[2].set_ylabel("")
ax[1].set_ylabel("")

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Plotting the computational complexity for the interactional agents over different amounts of turns

# Computational complexity for interactional agents
def f(turns, m):
    return (2 * m * (turns - 1)) + (2 * m * turns) + (2 * m)

# Specify different number of turns: stepsize is 2
turns = np.linspace(2, 68, 66)

# Different lexicon sizes (number of signals as maximum of signals and referents should be taken)
m = np.array([6, 15, 30])

# Specify colours for plotting
colors = ['#fdcc8a', '#fc8d59', '#d7301f']

# Plot the computational complexity for the different lexicon sizes
index = 0

for x in m:
    plt.plot(turns, f(turns, x), color=colors[index])
    index += 1

# Set labels and ticks and titles
plt.xlabel('Number of Turns', fontweight='bold', fontsize=18)
plt.ylabel('Computational Complexity', fontweight='bold', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
handles, labels = plt.gca().get_legend_handles_labels()
legend = plt.legend(handles, labels=["6x4", "15x10", "30x20"], title="Lexicon Size", fontsize=16)
plt.setp(legend.get_title(), fontsize=18, fontweight="bold")

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Computational complexity compared for the different agent types

# Specify the mean turns for the interactional agents
mean_turns = results_selected.groupby(["Number of Referents"])["Number of Turns"].mean()
mean_turns_list = [1.000000, 3.029633, 4.704133]

# Different computational complexity functions for the pragmatic agent types

# Frugal Pragmatic a
def g(m):
    return (16*m**2) + (4*m)

# Frugal Pragmatic b
def h(m):
    return (20*m**2) + (4*m)

# Fully Pragmatic
def i(m):
    return (20*m**2) + (2*m)

# Get results of the computational complexity functions based on lexicon sizes
results_comp = []
index = 0
for x in m:
    results_comp.extend([f(mean_turns_list[index], x), g(x), h(x), i(x)])
    index +=1

indices = [0,1,3,4,6,7,8,10,11]
sublist = [results_comp[i] for i in indices]

# Set height of bar
bars1 = sublist[:3]
bars2 = sublist[3:6]
bars3 = sublist[6:9]

# Set barwidth
barWidth = 0.25

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

fig, ax = plt.subplots()

colors = ['#fdcc8a', '#fc8d59', '#d7301f']

# Make the plot
ax.bar(r1, bars1, color=colors[0], width=barWidth, edgecolor='white', label='6x4')
ax.bar(r2, bars2, color=colors[1], width=barWidth, edgecolor='white', label='15x10')
ax.bar(r3, bars3, color=colors[2], width=barWidth, edgecolor='white', label='30x20')

# Set labels and ticks
plt.xlabel('Agent Type', fontweight='bold', fontsize=22)
plt.ylabel('Computational Complexity', fontweight='bold', fontsize=22)
plt.xticks([r + barWidth for r in range(len(bars1))], ["Interactional", "Frugally Pragmatic", "Fully Pragmatic"], fontsize=22)
plt.yticks(fontsize=22)
legend = plt.legend(title="Lexicon Size", fontsize=22)
plt.setp(legend.get_title(), fontsize=22, fontweight="bold")
ax.yaxis.grid(True, which='major')
ax.xaxis.grid(False, which='major')

plt.show()