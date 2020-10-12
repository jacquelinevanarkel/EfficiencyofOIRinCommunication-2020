# Efficiency of OIR in Communication

**Running the Simulations**

The folder *Implementation* contains the files needed to run the simulations:

  1. *Implementation.py*: the implementation of the interactional and pragmatic model.
  2. *lexicon_retriever.py*: used to retrieve lexicons.
  3. *lexicons (folder)*: folder containing lexicons needed to run the experiment.
  
Run the *implementation.py* file, while making sure the other files provided (*lexicon_retriever.py* and the lexicons) are available on your local machine. The results will be stored as csv files and zipped, which can be accessed again and analysed using the files contained in the folder *Visualising Results*:
  
1. *Reading_data.py*: in order to read in & preprocess the results from the simulations ran. Basically, it orders the results according to the order of reasoning of the agents. 
2. *plotting.py*: read in the results as preprocessed and stored in the *Reading_data.py* file. Then, numerous plotting functions are provided to explore the data.

**Exploring the Data**

If you wish to explore the original simulation data as described in the paper, this is made possible by providing the original simulation data [here] (https://osf.io/fxphv/). The data consist of six different files:

  1. *results.csv*: the complete results merged into this file.
 
  2. *results_interactional.csv*: contains the results of the interactional agents only.
  3. *results_pragmatic.csv*: contains the results of the pragmatic agents only. 
  4. *results_1.csv*: contains the results of the pragmatic agents with an order of pragmatic reasoning of 1 only.
  5. *results_frugal.csv*: contains the results of the pragmatic agents which started out with an order of pragmatic reasoning of 1 but went up to an order of 2 only.
  6. *results_2.csv*: contains the results of the pragmatic agents which started with an order of pragmatic reasoning of 2 only. 

According to your plotting wishes, one of these files can be selected. However, notice that the file *results.csv* contains both the interactional and pragmatic results, therefore *results_interactional.csv* and *results_pragmatic.csv* are its subsets. This is similar for the file *results_pragmatic.csv* and its subsets *results_1.csv*, *results_frugal.csv* and *results_2.csv*. 

Next, the *Visualising Results* folder contains the file *plotting.py* which can serve as an example to plot the data in different ways.
