# -------------------------------------- Part 1: RSA Implementation --------------------------------------

#Generate Lexicon
    #Code provided by Marieke

#Production
    #Input: dialogue history D, lexicon L, order n, intended referent r
    #Output: signal s, dialogue history D

    def production(D, L, n, r):

        #if n=0 & D is not empty
        L = conjunction(D, L)

        #calculate which signal s maximizes the probability, using the new lexicon if D was not empty

        return s, D



#Interpretation
    #Input: dialogue history D, Lexicon L, order n, observed signal s
    #Output: r, huh?, n+1, r (including recursion)
    #Calling functions:
        #conditional entropy over probability distributions
        #

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

#Simulate a Single Conversation

#Initializing Agents

#Lexicon

#Order of Pragmatic Reasoning

#Intention: ? --> randomly generated?