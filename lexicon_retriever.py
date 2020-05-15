import numpy as np
import pandas as pd

lexicons_df = pd.read_json('lexiconset.json') # Make sure to specify the correct path to the json file on your machine/the cluster

print("lexicons_df.head() is:")
with pd.option_context('display.max_columns', None): # this is in order to show all the columns (instead of a truncated version)
    print(lexicons_df.head())

print('')
print("lexicons_df.columns is:")
print(lexicons_df.columns)


def retrieve_lex(lexicons_df, n_signals, n_referents, ambiguity_level, n_lexicons):
    """
    Takes a pandas dataframe with columns 'vocabularySize', 'contextSize', 'ambiguity' and 'lexicons', and returns a numpy array containing a set of selected lexicons as specified by the other input arguments

    :param n_signals: int; desired number of signals
    :param n_referents: int; desired number of referents
    :param ambiguity_level: float (between 0.0 and 1.0); desired ambiguity level
    :param n_lexicons: int; desired number of lexicons
    :return: 3D numpy array; axis 0 = lexicons, axis 1 = signals, axis 2 = referents
    """
    selected_lexicons = lexicons_df[lexicons_df['vocabularySize'] == n_signals][lexicons_df['contextSize'] == n_referents][lexicons_df['ambiguity'] == ambiguity_level]['lexicons'].to_numpy()
    print('')
    print("selected_lexicons are:")
    with pd.option_context('display.max_columns', None):  # this is in order to show all the columns (instead of a truncated version)
        print(selected_lexicons)
    selected_lexicons = np.asarray(selected_lexicons[0])
    print('')
    print("selected_lexicons converted to numpy array are:")
    print(selected_lexicons)
    print("selected_lexicons.shape (i.e. dimensions of numpy array) is:")
    print(selected_lexicons.shape)
    random_indices = np.random.choice(np.arange(selected_lexicons.shape[0]), size=n_lexicons, replace=False) # chooses n_lexicons *unique* random integers in the range 0 to the number of lexicons in selected_lexicons
    print('')
    print("random_indices are:")
    print(random_indices)
    random_selection = selected_lexicons[random_indices]
    print("random_selection is:")
    print(random_selection)
    print("random_selection.shape (i.e. dimensions of numpy array) is:")
    print(random_selection.shape)
    return random_selection



my_lexicons = retrieve_lex(lexicons_df, 10, 8, 0.5, 5)


