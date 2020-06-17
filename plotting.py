# Necessary imports
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

# Read in data
results = pd.read_pickle("results.p")
results_interactional = pd.read_pickle("results_interactional.p")
results_pragmatic = pd.read_pickle("results_pragmatic.p")

# ------------------------------------------------- Plot the results ---------------------------------------------------
# # Plot the communicative success per lexicon size with lines for the chance levels & relative CS
# CS_means_interactional = results_interactional.groupby(["Number of Referents", "Ambiguity Level"])["Communicative Success"].mean()
# CS_means_pragmatic = results_pragmatic.groupby(["Number of Referents","Ambiguity Level"])["Communicative Success"].mean()
#
# # Calculate SDs
# SDs = []
# size = results_interactional.groupby(["Number of Referents", "Ambiguity Level"]).size()
# for index, value in CS_means_interactional.items():
#     P = value
#     N = size[index]
#     SD = ((P*(1-P))/N)**0.5
#     SDs.append(SD)
#
# plt.subplot(1, 2, 1)
# # set width of bar
# barWidth = 0.25
#
# # set height of bar
# bars1 = CS_means_interactional[:3]
# bars2 = CS_means_interactional[3:6]
# bars3 = CS_means_interactional[6:9]
#
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
#
# # Make the plot
# plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=[x * 2 for x in SDs[:3]])
# plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=[x * 2 for x in SDs[3:6]])
# plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=[x * 2 for x in SDs[6:9]])
#
# # Add xticks on the middle of the group bars
# plt.xlabel('Ambiguity Level', fontweight='bold')
# plt.ylabel('Communicative Success', fontweight='bold')
# plt.xticks([r + barWidth for r in range(len(bars1))], ['0.2','0.5','0.8'])
#
# plt.suptitle("Communicative Success for Different Ambiguity levels and Lexicon Sizes")
# plt.title("Interactional Agents")
#
# # Add chance levels
# chance_levels = (0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05)
# x_start = [r1[0]-(barWidth/2), r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(barWidth/2), r2[1]-(barWidth/2),
#            r3[1]-(barWidth/2), r1[2]-(barWidth/2), r2[2]-(barWidth/2), r3[2]-(barWidth/2)]
# x_end = [r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(1.5*barWidth), r2[1]-(barWidth/2), r3[1]-(barWidth/2),
#          r1[2]-(1.5*barWidth), r2[2]-(barWidth/2), r3[2]-(barWidth/2), r3[2]+barWidth-(barWidth/2)]
# plt.hlines(chance_levels, x_start, x_end)
#
# # Start second plot
# plt.subplot(1, 2, 2)
#
# # Calculate SDs
# SDs = []
# size = results_pragmatic.groupby(["Number of Referents", "Ambiguity Level"]).size()
# for index, value in CS_means_pragmatic.items():
#     P = value
#     N = size[index]
#     SD = ((P*(1-P))/N)**0.5
#     SDs.append(SD)
#
# # set width of bar
# barWidth = 0.25
#
# # set height of bar
# bars1 = CS_means_pragmatic[:3]
# bars2 = CS_means_pragmatic[3:6]
# bars3 = CS_means_pragmatic[6:9]
#
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
#
# # Make the plot
# plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=[x * 2 for x in SDs[:3]])
# plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=[x * 2 for x in SDs[3:6]])
# plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=[x * 2 for x in SDs[6:9]])
#
# # Add xticks on the middle of the group bars
# plt.xlabel('Ambiguity Level', fontweight='bold')
# plt.ylabel('Communicative Success', fontweight='bold')
# plt.xticks([r + barWidth for r in range(len(bars1))], ['0.2','0.5','0.8'])
#
# plt.title("Pragmatic Agents")
#
# # Add chance levels
# chance_levels = (0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05)
# x_start = [r1[0]-(barWidth/2), r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(barWidth/2), r2[1]-(barWidth/2),
#            r3[1]-(barWidth/2), r1[2]-(barWidth/2), r2[2]-(barWidth/2), r3[2]-(barWidth/2)]
# x_end = [r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(1.5*barWidth), r2[1]-(barWidth/2), r3[1]-(barWidth/2),
#          r1[2]-(1.5*barWidth), r2[2]-(barWidth/2), r3[2]-(barWidth/2), r3[2]+barWidth-(barWidth/2)]
# plt.hlines(chance_levels, x_start, x_end)
#
# # Create legend & Show graphic
# plt.legend(title="Lexicon Size")
# plt.show()
#
# # ---------------------------------------------------------------------------------------------------------------------
# # Second plot: number of turns interactional agents
# # Plot the communicative success per lexicon size with lines for the chance levels & relative CS
# turns_means_interactional = results_interactional.groupby(["Number of Referents", "Ambiguity Level"])["Number of Turns"].mean()
# turns_sd_interactional = results_interactional.groupby(["Number of Referents", "Ambiguity Level"])["Number of Turns"].std()
#
# # set width of bar
# barWidth = 0.25
#
# # set height of bar
# bars1 = turns_means_interactional[:3]
# bars2 = turns_means_interactional[3:6]
# bars3 = turns_means_interactional[6:9]
#
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
#
# # Make the plot
# plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=2*turns_sd_interactional[:3])
# plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=2*turns_sd_interactional[3:6])
# plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=2*turns_sd_interactional[6:9])
#
# # Add xticks on the middle of the group bars
# plt.xlabel('Ambiguity Level', fontweight='bold')
# plt.ylabel('Number of Turns', fontweight='bold')
# plt.xticks([r + barWidth for r in range(len(bars1))], ['0.2','0.5','0.8'])
#
# plt.title("Number of Turns for Different Ambiguity levels and Lexicon Sizes")
# plt.legend(title="Lexicon Size")
# plt.show()
#
# # ----------------------------------------------------------------------------------------------------------------------
# # Order of reasoning on CS
# CS_means = results.groupby(["Number of Referents", "Order of Reasoning Listener"])["Communicative Success"].mean()
#
# # Calculate SDs
# SDs = []
# size = results.groupby(["Number of Referents", "Order of Reasoning Listener"])["Communicative Success"].size()
# for index, value in CS_means.items():
#     P = value
#     N = size[index]
#     SD = ((P*(1-P))/N)**0.5
#     SDs.append(SD)
#
# # set width of bar
# barWidth = 0.25
#
# # set height of bar
# bars1 = CS_means[:3]
# bars2 = CS_means[3:6]
# bars3 = CS_means[6:9]
#
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
#
# # Make the plot
# plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=[x * 2 for x in SDs[:3]])
# plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=[x * 2 for x in SDs[3:6]])
# plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=[x * 2 for x in SDs[6:9]])
#
# # Add xticks on the middle of the group bars
# plt.xlabel('Order of Pragmatic Reasoning', fontweight='bold')
# plt.ylabel('Communicative Success', fontweight='bold')
# plt.ylim(0,1)
# plt.xticks([r + barWidth for r in range(len(bars1))], ['0','1','2'])
#
# # Add chance levels
# chance_levels = (0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05)
# x_start = [r1[0]-(barWidth/2), r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(barWidth/2), r2[1]-(barWidth/2),
#            r3[1]-(barWidth/2), r1[2]-(barWidth/2), r2[2]-(barWidth/2), r3[2]-(barWidth/2)]
# x_end = [r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(1.5*barWidth), r2[1]-(barWidth/2), r3[1]-(barWidth/2),
#          r1[2]-(1.5*barWidth), r2[2]-(barWidth/2), r3[2]-(barWidth/2), r3[2]+barWidth-(barWidth/2)]
# plt.hlines(chance_levels, x_start, x_end)
#
# plt.title("Communicative Success for Different Orders of Pragmatic Reasoning and Lexicon Sizes")
# plt.legend(title="Lexicon Size")
# plt.show()
#
# # ----------------------------------------------------------------------------------------------------------------------
# # Entropy threshold and lexicon size
# CS_means = results_interactional.groupby(["Number of Referents", "Entropy Threshold"])["Communicative Success"].mean()
# CS_means_pragmatic = results_pragmatic.groupby(["Number of Referents", "Entropy Threshold"])["Communicative Success"].mean()
# turns_means = results.groupby(["Number of Referents", "Entropy Threshold"])["Number of Turns"].mean()
# turns_sd = results.groupby(["Number of Referents", "Entropy Threshold"])["Number of Turns"].std()
#
# plt.subplot(1, 2, 1)
#
# # Calculate SDs
# SDs = []
# size = results_interactional.groupby(["Number of Referents", "Entropy Threshold"])["Communicative Success"].size()
# for index, value in CS_means.items():
#     P = value
#     N = size[index]
#     SD = ((P*(1-P))/N)**0.5
#     SDs.append(SD)
#
# # set width of bar
# barWidth = 0.25
#
# # set height of bar
# bars1 = CS_means[:3]
# bars2 = CS_means[3:6]
# bars3 = CS_means[6:9]
#
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
#
# # Make the plot
# plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=[x * 2 for x in SDs[:3]])
# plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=[x * 2 for x in SDs[3:6]])
# plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=[x * 2 for x in SDs[6:9]])
#
# # Add xticks on the middle of the group bars
# plt.xlabel('Entropy Threshold', fontweight='bold')
# plt.ylabel('Communicative Success', fontweight='bold')
# plt.ylim(0,1)
# plt.xticks([r + barWidth for r in range(len(bars1))], ['0.8','1.0','1.5'])
#
# # Add chance levels
# chance_levels = (0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05)
# x_start = [r1[0]-(barWidth/2), r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(barWidth/2), r2[1]-(barWidth/2),
#            r3[1]-(barWidth/2), r1[2]-(barWidth/2), r2[2]-(barWidth/2), r3[2]-(barWidth/2)]
# x_end = [r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(1.5*barWidth), r2[1]-(barWidth/2), r3[1]-(barWidth/2),
#          r1[2]-(1.5*barWidth), r2[2]-(barWidth/2), r3[2]-(barWidth/2), r3[2]+barWidth-(barWidth/2)]
# plt.hlines(chance_levels, x_start, x_end)
#
# plt.suptitle("Communicative Success for Different Entropy Thresholds and Lexicon Sizes")
# plt.title("Interactional Agents")
# plt.legend(title="Lexicon Size")
#
# plt.subplot(1, 2, 2)
#
# # Calculate SDs
# SDs = []
# size = results_pragmatic.groupby(["Number of Referents", "Entropy Threshold"])["Communicative Success"].size()
# for index, value in CS_means_pragmatic.items():
#     P = value
#     N = size[index]
#     SD = ((P*(1-P))/N)**0.5
#     SDs.append(SD)
#
# # set width of bar
# barWidth = 0.25
#
# # set height of bar
# bars1 = CS_means_pragmatic[:3]
# bars2 = CS_means_pragmatic[3:6]
# bars3 = CS_means_pragmatic[6:9]
#
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
#
# # Make the plot
# plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=[x * 2 for x in SDs[:3]])
# plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=[x * 2 for x in SDs[3:6]])
# plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=[x * 2 for x in SDs[6:9]])
#
# # Add xticks on the middle of the group bars
# plt.xlabel('Entropy Threshold', fontweight='bold')
# plt.ylabel('Communicative Success', fontweight='bold')
# plt.ylim(0,1)
# plt.xticks([r + barWidth for r in range(len(bars1))], ['0.8','1.0','1.5'])
#
# # Add chance levels
# chance_levels = (0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05)
# x_start = [r1[0]-(barWidth/2), r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(barWidth/2), r2[1]-(barWidth/2),
#            r3[1]-(barWidth/2), r1[2]-(barWidth/2), r2[2]-(barWidth/2), r3[2]-(barWidth/2)]
# x_end = [r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(1.5*barWidth), r2[1]-(barWidth/2), r3[1]-(barWidth/2),
#          r1[2]-(1.5*barWidth), r2[2]-(barWidth/2), r3[2]-(barWidth/2), r3[2]+barWidth-(barWidth/2)]
# plt.hlines(chance_levels, x_start, x_end)
#
# plt.title("Pragmatic Agents")
# plt.legend(title="Lexicon Size")
# plt.show()
# # ----------------------------------------------------------------------------------------------------------------------
# # set width of bar
# barWidth = 0.25
#
# # set height of bar
# bars1 = turns_means[:3]
# bars2 = turns_means[3:6]
# bars3 = turns_means[6:9]
#
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
#
# # Make the plot
# plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=2*turns_sd[:3])
# plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=2*turns_sd[3:6])
# plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=2*turns_sd[6:9])
#
# # Add xticks on the middle of the group bars
# plt.xlabel('Entropy Threshold', fontweight='bold')
# plt.ylabel('Number of Turns', fontweight='bold')
# plt.xticks([r + barWidth for r in range(len(bars1))], ['0.8','1.0','1.5'])
#
# plt.title("Number of Turns for Different Entropy Thresholds and Lexicon Sizes")
# plt.legend(title="Lexicon Size")
# plt.show()
#
# # ----------------------------------------------------------------------------------------------------------------------
# # Entropy over time (turns)
#
# # ----------------------------------------------------------------------------------------------------------------------
# # Order of reasoning - where listener ended
#
# # Order of reasoning on CS
# results["Order of Reasoning"] = np.where(results["Reached Threshold Order"]==True, 2, results["Order of Reasoning Listener"])
# CS_means = results.groupby(["Number of Referents", "Order of Reasoning"])["Communicative Success"].mean()
#
# # Calculate SDs
# SDs = []
# size = results.groupby(["Number of Referents", "Order of Reasoning"])["Communicative Success"].size()
# for index, value in CS_means.items():
#     P = value
#     N = size[index]
#     SD = ((P*(1-P))/N)**0.5
#     SDs.append(SD)
#
# plt.subplot(1, 2, 1)
# # set width of bar
# barWidth = 0.25
#
# # set height of bar
# bars1 = CS_means[:3]
# bars2 = CS_means[3:6]
# bars3 = CS_means[6:9]
#
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
#
# # Make the plot
# plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4', yerr=[x * 2 for x in SDs[:3]])
# plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10', yerr=[x * 2 for x in SDs[3:6]])
# plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20', yerr=[x * 2 for x in SDs[6:9]])
#
# # Add xticks on the middle of the group bars
# plt.xlabel('Order of Pragmatic Reasoning', fontweight='bold')
# plt.ylabel('Communicative Success', fontweight='bold')
# plt.ylim(0,1)
# plt.xticks([r + barWidth for r in range(len(bars1))], ['0','1','2'])
#
# # Add chance levels
# chance_levels = (0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05)
# x_start = [r1[0]-(barWidth/2), r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(barWidth/2), r2[1]-(barWidth/2),
#            r3[1]-(barWidth/2), r1[2]-(barWidth/2), r2[2]-(barWidth/2), r3[2]-(barWidth/2)]
# x_end = [r2[0]-(barWidth/2), r3[0]-(barWidth/2), r1[1]-(1.5*barWidth), r2[1]-(barWidth/2), r3[1]-(barWidth/2),
#          r1[2]-(1.5*barWidth), r2[2]-(barWidth/2), r3[2]-(barWidth/2), r3[2]+barWidth-(barWidth/2)]
# plt.hlines(chance_levels, x_start, x_end)
#
# plt.title("Communicative Success \n for Different Orders of Pragmatic Reasoning and Lexicon Sizes")
# plt.legend(title="Lexicon Size")
#
# plt.subplot(1, 2, 2)
# # set width of bar
# barWidth = 0.25
#
# # set height of bar
# bars1 = size[:3]
# bars2 = size[3:6]
# bars3 = size[6:9]
#
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
#
# # Make the plot
# plt.bar(r1, bars1, color='#ABABFF', width=barWidth, edgecolor='white', label='6x4')
# plt.bar(r2, bars2, color='#5757DF', width=barWidth, edgecolor='white', label='15x10')
# plt.bar(r3, bars3, color='#000077', width=barWidth, edgecolor='white', label='30x20')
#
# # Add xticks on the middle of the group bars
# plt.xlabel('Order of Pragmatic Reasoning', fontweight='bold')
# plt.ylabel('Amount of Interactions', fontweight='bold')
# plt.ylim(0,480000)
# plt.xticks([r + barWidth for r in range(len(bars1))], ['0','1','2'])
#
# plt.title("Amount of Interactions \n for Different Orders of Pragmatic Reasoning and Lexicon Sizes")
# plt.legend(title="Lexicon Size")
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# results["Entropy"] = np.where(len(results["Entropy"])==30, results["Entropy"], list(itertools.chain(results["Entropy"], [np.NAN]*(30-len(results["Entropy"])))))
#
# print(results["Entropy"][:30])

#sns.lineplot(x=np.arange(30),y="Entropy",hue="Order of Reasoning Listener",data=results)

# ----------------------------------------------------------------------------------------------------------------------
# CS interactional vs pragmatic
fig, ax = plt.subplots(1,2)
ax[0].hlines(y=[0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05],xmin=[-0.4, -0.13, 0.13, 0.60, 0.87, 1.13, 1.60, 1.87, 2.13],
           xmax=[-0.13, 0.13, 0.40, 0.87, 1.13, 1.40, 1.87, 2.13, 2.40])
sns.barplot(x="Ambiguity Level", y="Communicative Success", hue="Number of Referents", data=results_interactional, ax=ax[0])
plt.suptitle("Mean Communicative Success for Different Ambiguity levels and Lexicon Sizes")
ax[0].set_ylabel("Mean Communicative Success")
ax[0].set_title("Interactional Agents")
ax[0].set_ylim(0,1)
ax[1].hlines(y=[0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05],xmin=[-0.4, -0.13, 0.13, 0.60, 0.87, 1.13, 1.60, 1.87, 2.13],
           xmax=[-0.13, 0.13, 0.40, 0.87, 1.13, 1.40, 1.87, 2.13, 2.40])
sns.barplot(x="Ambiguity Level", y="Communicative Success", hue="Number of Referents", data=results_pragmatic, ax=ax[1])
ax[1].set_title("Pragmatic Agents")
ax[1].set_ylabel("Mean Communicative Success")
ax[1].set_ylim(0,1)
plt.show()

# Number of referents, ambiguity level, number of turns
sns.violinplot(x="Ambiguity Level", y="Number of Turns", hue="Number of Referents", data=results_interactional)
plt.title("Number of Turns for Different Ambiguity levels and Lexicon Sizes")
plt.show()

# CS by order of reasoning, number of referents
plt.title("Communicative Success for Different Orders of Pragmatic Reasoning \n (where Listener Ended) and Lexicon Sizes")
results["Order of Reasoning"] = np.where(results["Reached Threshold Order"]==True, 2, results["Order of Reasoning Listener"])
sns.barplot(x="Order of Reasoning", y="Communicative Success", hue="Number of Referents", data=results)
plt.ylim(0,1)
plt.show()

# CS by entropy threshold and lexicon size pragmatic vs interactional
fig, ax = plt.subplots(1,2)
ax[0].hlines(y=[0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05],xmin=[-0.4, -0.13, 0.13, 0.60, 0.87, 1.13, 1.60, 1.87, 2.13],
           xmax=[-0.13, 0.13, 0.40, 0.87, 1.13, 1.40, 1.87, 2.13, 2.40])
sns.barplot(x="Entropy Threshold", y="Communicative Success", hue="Number of Referents", data=results_interactional, ax=ax[0])
plt.suptitle("Mean Communicative Success for Different Entropy Thresholds and Lexicon Sizes")
ax[0].set_ylabel("Mean Communicative Success")
ax[0].set_ylim(0,1)
ax[0].set_title("Interactional Agents")
ax[1].hlines(y=[0.25, 0.1, 0.05, 0.25, 0.1, 0.05, 0.25, 0.1, 0.05],xmin=[-0.4, -0.13, 0.13, 0.60, 0.87, 1.13, 1.60, 1.87, 2.13],
           xmax=[-0.13, 0.13, 0.40, 0.87, 1.13, 1.40, 1.87, 2.13, 2.40])
sns.barplot(x="Entropy Threshold", y="Communicative Success", hue="Number of Referents", data=results_pragmatic, ax=ax[1])
ax[1].set_title("Pragmatic Agents")
ax[1].set_ylabel("Mean Communicative Success")
ax[1].set_ylim(0,1)
plt.show()

# Turns for entropy and lexicon size
sns.violinplot(x="Entropy Threshold", y="Number of Turns", hue="Number of Referents", data=results_interactional)
plt.title("Number of Turns for Different Entropy Thresholds \n and Lexicon Sizes (Interactional)")
plt.show()

# CS for different ambiguity levels and entropy levels per lexicon size
lex1 = results[results["Number of Referents"] == 4]
lex2 = results[results["Number of Referents"] == 10]
lex3 = results[results["Number of Referents"] == 20]
data1 = pd.pivot_table(lex1, values='Communicative Success', index=['Entropy Threshold'], columns='Ambiguity Level')
data2 = pd.pivot_table(lex2, values='Communicative Success', index=['Entropy Threshold'], columns='Ambiguity Level')
data3 = pd.pivot_table(lex3, values='Communicative Success', index=['Entropy Threshold'], columns='Ambiguity Level')
fig, ax = plt.subplots(1,3)
sns.heatmap(data1, ax=ax[0], vmin=0, vmax=1, cmap="YlGnBu")
sns.heatmap(data2, ax=ax[1], vmin=0, vmax=1, cmap="YlGnBu")
sns.heatmap(data3, ax=ax[2], vmin=0, vmax=1, cmap="YlGnBu")
plt.suptitle("Mean Communicative Success for Different Ambiguity Levels and Entropy Levels per Lexicon Sizee")
ax[0].set_title("Lexicon: 6x4")
ax[1].set_title("Lexicon: 15x10")
ax[2].set_title("Lexicon: 30x20")
plt.show()