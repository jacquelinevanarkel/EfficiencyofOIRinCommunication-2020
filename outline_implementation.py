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

    #THINK ABOUT THIS!!!!
    def conjunction(D, L):

        return L



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

#Order of Pragmatic Reasoning

#Intention: ? --> randomly generated?