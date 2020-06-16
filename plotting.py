# Necessary imports
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read in data
results = pd.read_pickle("results.p")
results_interactional = pd.read_pickle("results_interactional.p")
results_pragmatic = pd.read_pickle("results_pragmatic.p")

# ------------------------------------------------- Plot the results ---------------------------------------------------
# Plot the communicative success per lexicon size with lines for the chance levels & relative CS
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
# Second plot: number of turns interactional agents
# Plot the communicative success per lexicon size with lines for the chance levels & relative CS
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

plt.title("Number of Turns for Different Ambiguity levels and Lexicon Sizes")
plt.legend(title="Lexicon Size")
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Order of reasoning on CS
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

plt.title("Communicative Success for Different Orders of Pragmatic Reasoning and Lexicon Sizes")
plt.legend(title="Lexicon Size")
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Entropy threshold and lexicon size
CS_means = results_interactional.groupby(["Number of Referents", "Entropy Threshold"])["Communicative Success"].mean()
CS_means_pragmatic = results_pragmatic.groupby(["Number of Referents", "Entropy Threshold"])["Communicative Success"].mean()
turns_means = results.groupby(["Number of Referents", "Entropy Threshold"])["Number of Turns"].mean()
turns_sd = results.groupby(["Number of Referents", "Entropy Threshold"])["Number of Turns"].std()

plt.subplot(1, 2, 1)

# Calculate SDs
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

plt.title("Pragmatic Agents")
plt.legend(title="Lexicon Size")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
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
# Entropy over time (turns)
