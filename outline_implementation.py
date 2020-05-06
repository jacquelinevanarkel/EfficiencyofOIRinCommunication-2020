# Necessary imports
import pandas as pd

# -------------------------------------- Part 1: RSA Implementation --------------------------------------

#Generate Lexicon
    #Code provided by Marieke

# /////////// Production ///////////
    #Input: dialogue history D, lexicon L, order n, intended referent r
    #Output: signal s, dialogue history D

    def production(D, L, n, r):

        #if n=0 & D is not empty
        L = conjunction(D, L)

        #calculate which signal s maximizes the probability, using the new lexicon if D was not empty

        return s, D


# /////////// Interpretation ///////////
    #Input: dialogue history D, Lexicon L, order n, observed signal s
    #Output: r, huh?, n+1, r (including recursion)
    #Calling functions:
        #conditional entropy over probability distributions
        #itself
        #production: as means of OIR

    def listener(D, L, n, s, n_t):
        # if n = 0 and D is not empty
        s = conjunction(D, s)

        H = conditional_entropy(r, s, L)

        # when entropy low or when n = n_t --> recursion till n=0
        act = interpretation(D, L, n, s)

        # if entropy high
        # if n < n_t
        act = interpretation(D, L, n + 1, s)
        # else: turn to speaker --> produces new signal
        act = production(D, L, 0, i)

        return act

    def interpretation(D, L, n, s, i):
        #if n = 0 and D is not empty
        s = conjunction(D, s)

        H = conditional_entropy(r, s, L)

        #when entropy low or when n = n_t --> recursion till n=0
        interpretation(D, L, n, s)

        #if entropy high
            #if n < n_t
        interpretation(D, L, n+1, s)
            #else: turn to speaker --> produces new signal
        production(D, L, 0, i)


        return r

    def conditional_entropy(r, s, L):

        return H

    #Depends on how the lexicon is structured: as a pandas dataframe? --> change accordingly
    #So this is not correct yet, I stopped in the middle as I think it would be useful to first see the structure
    #of the lexicon before defining this 
    def conjunction(D, L = None, s = None):
        if L is not None:
            for index, row in L.iterrows():
                for s_old in D:
                    for x in L.iloc:
                        if s_old == 1 & L.iloc[x] == 1:
                            L_new[x] = 1
                        else
                            L_new[x] = 0
        else:
            for s_old in D:
                    if s_old == 1 & s == 1:
                        L_new[x] = 1
                    else
                        L_new[x] = 0
        return L_new



# -------------------------------------- Part 2: Simulation --------------------------------------
#                                   Simulate a Single Conversation

#Initializing Agents: order of pragmatic reasoning, ambiguity level lexicon, type (listener, speaker), optional: entropy threshold
class agent(Agent):
    #Initialization
    def __init__(self, order, ambiguity_level, agent_type):
        self.n = order
        self.a = ambiguity_level
        self.type = agent_type

Agent1 = agent(0, 1, "speaker")
Agent2 = agent(0, 1, "listener")

#Lexicon

#Intention: ? --> randomly generated?