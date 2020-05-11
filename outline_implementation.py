# Necessary imports
import pandas as pd
import numpy as np
import lexicon_retriever as lex_retriever

# --------------------------------------------- Part 1: RSA Implementation ---------------------------------------------
# ///////////////////////////////////////////////////// Production /////////////////////////////////////////////////////
class production:
#Input: dialogue history D, lexicon L, order n, intended referent i
#Output: signal s, dialogue history D

    # Initialization
    def __init__(self, lexicon, intention, n=0, dialogueHistory=None):
        self.L = lexicon
        self.i = intention
        self.n = n
        self.D = dialogueHistory

    def produce(self):
        if n = 0:
            production_literal(self)
        else:
            production_pragmatic(self)

    def production_literal(self):
        #if D is not empty
        L = conjunction(self)

        #calculate which signal s maximizes the probability, using the new lexicon if D not empty
        # --> call function!

        #update D
        return s, D

    def production_pragmatic(self):
        #calculate which signal s maximizes the probability --> call function

        return s

    def conjunction(self):

        newLexicon = np.zeros(self.L.shape)
        index_signal = 0

        for signal in self.L:
            index_referent = 0
            for r, r2 in zip(self.D[:-1], signal):
                if r == 1 & r2 == 1:
                    newLexicon[index_signal, index_referent] = 1
                index_referent += 1
            index_signal +=1

        return newLexicon


# /////////////////////////////////////////////////// Interpretation ///////////////////////////////////////////////////
class interpretation:
    #Input: dialogue history D, Lexicon L, order n, observed signal s, intended referent i (in case of production (?))
    #       threshold pragmatic reasoning n_t, turns, entropy threshold
    #Output: [r, huh? (--> production), n+1, r (including recursion)] --> intended referent
    #Calling functions:
        #conditional entropy over probability distributions
        #conjunction
        #production: as means of OIR

    # Initialization
    def __init__(self, lexicon, signal, n_t = 2, n = 0, dialogueHistory=None, entropyThreshold):
        self.L = lexicon
        self.s = signal
        self.n_t = n_t
        self.n = n
        self.D = dialogueHistory
        self.H_t = entropyThreshold

    def interpret(self):
        if n = 0:
            interpretation_literal(self)
        else:
            interpretation_pragmatic(self)

    def interpretation_literal(self):
        # if D is not empty
        s = conjunction(self)

        #calculate posterior distribution given s and D

        #calculate entropy of posterior distribution
        H = conditional_entropy(r, self.s, self.L)

        # when H < H_t: output inferred referent
        # output = referent --> call function

        # when H > H_t:
        # turn to speaker --> output = OIR
        output = "OIR"

        return output

    def interpretation_pragmatic(self):
        #calculate posterior distribution given s and L

        #calculate the entropy of posterior distribution
        H = conditional_entropy(r, self.s, self.L)

        #if H > H_t & n < n_t
        interpretation(self.L, self.n+1, self.s, self.n_t)

        # if H < H_t or when n = n_t --> recursion till n=0
        #save when n_t is reached!
        #interpretation(D, L, n, s) --> call function!

        return r

    def conditional_entropy(self, r):

        return H

    def conjunction(self):
        combined_signals = np.zeros(L.shape[1])
        index = 0
        for r, r2 in zip(self.D[:-1], self.s):
            if r == 1 & r2 ==1:
                combined_signals[index] = 1
            index += 1

        return combined_signals

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# You either have a signal + signal (L) or Lexicon + signal (S)
# def conjunction(D, L = None, s = None):
#     if L is not None:
#         for index, row in L.iterrows():
#             for s_old in D:
#                 for x in L.iloc:
#                     if s_old == 1 & L.iloc[x] == 1:
#                         L_new[x] = 1
#                     else
#                         L_new[x] = 0
#     else:
#         for s_old in D:
#                 if s_old == 1 & s == 1:
#                     L_new[x] = 1
#                 else
#                     L_new[x] = 0
#     return L_new



# ------------------------------------------------- Part 2: Simulation -------------------------------------------------
#                                            Simulate a Single Conversation

#Initializing Agents: order of pragmatic reasoning, ambiguity level lexicon, type (listener, speaker), optional: entropy threshold
class agent:
    #Initialization
    def __init__(self, order, lexicon, agentType, entropyThreshold):
        self.n = order
        self.L = lexicon
        self.type = agentType
        self.H_t = entropyThreshold

Agent1 = agent(0, 1, L, "speaker", H_t)
Agent2 = agent(0, 1, L, "listener", H_t)

#Generate Lexicons
lexicons_df = pd.read_json('lexiconset.json')
# 5 lexicons with 10 signals, 8 referents, and an ambiguity level of 0.5
n_signals = 10
n_referents = 8
ambiguity_level = 0.5
n_lexicons = 5
lexicons = lex_retriever.retrieve_lex(lexicons_df, n_signals, n_referents, ambiguity_level, n_lexicons)

#Intention: randomly generated from uniform distribution
intention = np.random.randint(n_referents+1)

#One interaction/Communication: implementation not done yet!!!
def interaction(agent1, agent2):
    #Initialize this with column names
    output = pd.DataFrame()
    turns = 0
    if agent1.type == "speaker":
        producedSignal = production().produce()
        turns += 1
        listenerOutput = interpretation().interpret()
        turns += 1
        while listener_output == "OIR":
            producedSignal = production().produce()
            turns += 1
            listenerOutput = interpretation().interpret()
            turns += 1
    else:
        #agent2 produces
        producedSignal = production().produce()
        turns += 1
        listenerOutput = interpretation().interpret()
        turns += 1
        while listenerOutput == "OIR":
            producedSignal = production().produce()
            turns += 1
            listenerOutput = interpretation().interpret()
            turns += 1
    output = output.append(producedSignal, listenerOutput, turns, agent1.n)
    return output

# ///////////////////////////////////////// Measurements: dependent variables /////////////////////////////////////////
#Communicative success
def communicativeSuccess(i, r):
    if i == r:
        cs = 1
    else:
        cs = 0

    return cs

def averageComSuc():
    sum = 0
    for i, r in interactions:
        sum += communicative_success(i, r)

    return sum / len(interactions)

#Complexity: also a measurement, but not included here

# //////////////////////////////////////////////// Running Simulations ////////////////////////////////////////////////
def simulation(n_interactions, lexicon, ambiguityLevel):

    return dataframe
