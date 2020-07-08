# Efficiency of OIR in Communication

**Running the simulations**

The folder 'Implementation' contains the files needed to run the simulations:
  1. Implementation.py: the implementation of the interactional and pragmatic model.
  2. lexicon_retriever.py: used to retrieve lexicons.
  3. lexicons (folder): folder containing lexicons needed to run the experiment.
  
Run the 'implementation.py' file, while making sure the other files provided (lexicon_retriever.py and the lexicons) are available on your local machine. The results will be pickled, which can be accessed again and analysed using the files contained in the folder 'Visualising Results':
  1. Reading_data.py: in order to read in & preprocess the results from the simulations ran. Basically, it orders the results according to the order of reasoning of                       the agents. 
  2. plotting.py: read in the results as preprocessed and stored in the 'Reading_data.py' file. Then, numerous plotting functions are provided to explore the data. 
  
It is important to take into account that reading in the data can take a lot of time as the data files are quite big.

**Exploring the Data**

If you wish to explore the original simulation data as described in the paper, this is made possible by the pickled results in the 'Results' folder. This folder contains three different files:
  1. results.p: the complete results merged into this file.
  2. results_interactional.p: contains the results of the interactional agents only.
  3. results_pragmatic.p: contains the results of the pragmatic agents only. 

According to your plotting wishes, one of these files can be selected. However, notice that the file 'results.p' contains both the interactional and pragmatic results, therefore 'results_interactional.p' and 'results_pragmatic.p' are its subsets.

Next, the 'Visualising Results' folder contains the file 'plotting.py' which can serve as an example to plot the data in different ways. 
