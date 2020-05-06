# Necessary imports
import pandas as pd

# -------------------------------------- Part 1: RSA Implementation --------------------------------------

#Generate Lexicon
    #Code provided by Marieke

# ////////////////////////// Production //////////////////////////
    #Input: dialogue history D, lexicon L, order n, intended referent r, turns
    #Output: signal s, dialogue history D, turns

    def production(L, n, r, D = None, turns = 0):
        #if n=0 & D is not empty
        L = conjunction(D, L)

        #calculate which signal s maximizes the probability, using the new lexicon if D not empty

        #update D

        turns += 1
        return s, D, turns


# ////////////////////////// Interpretation //////////////////////////
    class interpretation:
        #Input: dialogue history D, Lexicon L, order n, observed signal s, intended referent i (in case of production (?))
        #       threshold pragmatic reasoning n_t, turns
        #Output: [r, huh? (--> production), n+1, r (including recursion)] --> intended referent
        #Calling functions:
            #conditional entropy over probability distributions
            #conjunction
            #production: as means of OIR

        # Initialization
        def __init__(self, lexicon, signal, intention, n_t = 2, n = 0):
            self.L = lexicon
            self.s = signal
            self.i = intention
            self.n_t = n_t
            self.n = n

        def interpret(self, D = none):
            if n = 0:
                interpretation_literal(D, L, s, i)
            else:
                interpretation_pragmatic(L, n, s, n_t)

        def interpretation_literal(D, L, s, i, turns):
            # if D is not empty
            s = conjunction(D, s)

            #calculate posterior distribution given s and D

            #calculate entropy of posterior distribution
            H = conditional_entropy(r, s, L)

            # when H low: output inferred referent
            # output = referent

            # when H high:
            # turn to speaker --> produces new signal
            turns += 1
            s, D, turns = production(L, 0, i, D, turns)
            output = interpretation_literal(D, L, s, i, turns)

            turns += 1
            return output, turns

        def interpretation_pragmatic(L, n, s, n_t):
            #calculate posterior distribution given s and L

            #calculate the entropy of posterior distribution
            H = conditional_entropy(r, s, L)

            #if entropy high & n < n_t
            interpretation(L, n+1, s, n_t)

            # when entropy low or when n = n_t --> recursion till n=0
            #interpretation(D, L, n, s)

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
class agent:
    #Initialization
    def __init__(self, order, ambiguity_level, agent_type):
        self.n = order
        self.a = ambiguity_level
        self.type = agent_type

Agent1 = agent(0, 1, "speaker")
Agent2 = agent(0, 1, "listener")

#Lexicon

#Intention: ? --> randomly generated?

#Communicative success
def communicative_success(i, r):
    if i == r:
        cs = 1
    else:
        cs = 0

    return cs

def average_com_suc():
    sum = 0
    for i, r in interactions:
        sum += communicative_success(i, r)

    return sum / len(interactions)
