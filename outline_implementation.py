# Necessary imports
import pandas as pd

# -------------------------------------- Part 1: RSA Implementation --------------------------------------

#Generate Lexicon
    #Code provided by Marieke

# ////////////////////////// Production //////////////////////////
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
            production_literal(L, i, D)
        else:
            production_pragmatic(n)

    def production_literal(self, L, i, D):
        #if D is not empty
        L = conjunction(D, L)

        #calculate which signal s maximizes the probability, using the new lexicon if D not empty

        #update D
        return s, D

    def production_pragmatic(self, n):
        #calculate which signal s maximizes the probability

        return s


# ////////////////////////// Interpretation //////////////////////////
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
            interpretation_literal(D, L, s)
        else:
            interpretation_pragmatic(L, n, s, n_t)

    def interpretation_literal(D, L, s):
        # if D is not empty
        s = conjunction(D, s)

        #calculate posterior distribution given s and D

        #calculate entropy of posterior distribution
        H = conditional_entropy(r, s, L)

        # when H < H_t: output inferred referent
        # output = referent

        # when H > H_t:
        # turn to speaker --> output = OIR
        output = "OIR"

        return output

    def interpretation_pragmatic(L, n, s, n_t):
        #calculate posterior distribution given s and L

        #calculate the entropy of posterior distribution
        H = conditional_entropy(r, s, L)

        #if H > H_t & n < n_t
        interpretation(L, n+1, s, n_t)

        # if H < H_t or when n = n_t --> recursion till n=0
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
    def __init__(self, order, lexicon, agentType, entropyThreshold):
        self.n = order
        self.L = lexicon
        self.type = agentType
        self.H_t = entropyThreshold

Agent1 = agent(0, 1, L, "speaker")
Agent2 = agent(0, 1, L, "listener")

#Lexicon: include ambiguity level

#Intention: randomly generated

#One interaction/Communication
def interaction(agent1, agent2):
    turns = 0
    if agent1.type = "speaker":
        production().produce()
        turns += 1
        interpretation().interpret()
    else:
        #agent2 produces
        production().produce()
        turns += 1
        listener_output = interpretation().interpret()
        if listener_output == "OIR":
            turns += 1
            production().produce()
        else:
            break
    return turns

# ////////////////////////// Measurements: dependent variables //////////////////////////
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

# ////////////////////////// Running Simulations //////////////////////////
def simulation(n_interactions, lexicon, ambiguityLevel):

    return dataframe
