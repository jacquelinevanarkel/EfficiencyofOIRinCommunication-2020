# Necessary imports
import pandas as pd
import numpy as np
import lexicon_retriever as lex_retriever
import math

# --------------------------------------------- Part 1: RSA Implementation ---------------------------------------------
# ///////////////////////////////////////////////////// Production /////////////////////////////////////////////////////
class Production:

    def __init__(self, lexicon, intention, order, dialogue_history):
        """
        Initialization of class.
        :param lexicon: array; the lexicon used by the speaker
        :param intention: int; the intended referent (index) meant by the speaker
        :param order: int; the order of pragmatic reasoning of the speaker
        :param dialogue_history: list; the previously produced signals
        """

        self.lexicon = lexicon
        self.intention = intention
        self.order = order
        self.dialogue_history = dialogue_history

        self.norm_lex = self.normalise_lexicon()

    def produce(self):
        """
        Start producing by calling the function corresponding to the order of pragmatic reasoning of the speaker.
        :return: int; signal by calling the corresponding production function.
        """

        if self.order == 0:
            return self.production_literal()
        else:
            return self.production_pragmatic()

    def production_literal(self):
        """
        For literal speakers, a signal is produced by choosing the signal that maximizes the probability given the
        intention, making use of the dialogue history if provided.
        :return: int; signal
        """

        #Perform conjunction if a dialogue history is available: the lexicon changes accordingly for this interaction
        if self.dialogue_history:
            self.lexicon = self.conjunction()

        #Calculate which signal maximizes the probability, given the intention
        norm_lex = self.normalise_lexicon()
        max_signal = np.amax(norm_lex, axis=0)[self.intention]
        indices = np.where(np.transpose(norm_lex)[self.intention] == max_signal)
        signal = int(np.random.choice(indices[0], 1, replace=False))

        return int(signal)

    def production_pragmatic(self):
        """
        For pragmatic speakers, a signal is produced by choosing the signal that maximizes the probability given the
        intention, determined by the order of pragmatic reasoning.
        :return: int; signal
        """

        #calculate which signal maximizes the probability
        prob_lex = np.zeros(np.size(self.lexicon, 0))
        for signal_index in range(np.size(self.lexicon, 0)):
            prob_lex[signal_index] = self.prob_signal_referent(self.order, self.intention, signal_index)

        max_prob_signal = np.amax(prob_lex, axis=0)
        indices = np.where(prob_lex == max_prob_signal)
        signal = int(np.random.choice(indices[0], 1))

        return signal

    def conjunction(self):
        """
        Perform conjunction between the lexicon and the last produced signal.
        :return: array; new lexicon based on the conjunction
        """

        #Initialize new lexicon
        new_lexicon = np.zeros(self.lexicon.shape)
        index_signal = 0

        #Perform conjunction on last produced signal and the lexicon
        for signal in self.lexicon:
            index_referent = 0
            for ref, ref2 in zip(self.lexicon[self.dialogue_history[-1]], signal):
                if ref == 1 and ref2 == 1:
                    new_lexicon[index_signal, index_referent] = 1
                index_referent += 1
            index_signal +=1

        return new_lexicon

    def normalise_lexicon(self):
        """
        Calculate the probabilities of the signals given the referents to create a lexicon filled with probabilities.
        :return: array; lexicon with probabilities of signals given referents
        """

        prob_lex_production = np.divide(self.lexicon, np.sum(self.lexicon, axis=0), out=np.zeros_like(self.lexicon),
                                        where=np.sum(self.lexicon, axis=0)!=0)

        return prob_lex_production

    def prob_signal_referent(self, order, referent_index, signal_index):
        """
        Calculate the probability of the signal given the referent: probability of the referent given the signal
        (listener: same order of pragmatic reasoning) divided by the sum of the (listener) probabilities of the referent
        given all signals. If n = 0, calculate the literal probability.
        :param order: int; the order of pragmatic reasoning of the speaker
        :param referent_index: int; the index of the referent in the lexicon
        :param signal_index: int; the index of the signal in the lexicon
        :return: float; the probability of the signal given the referent
        """

        #When the order is 0, the literal probability is calculated
        if order == 0:
            return self.norm_lex[signal_index][referent_index]
        #When the order is not 0, the pragmatic probabilities are calculated
        else:
            sum_prob_referent = 0.0
            for signal_index_set in range(np.size(self.lexicon, 0)):
                sum_prob_referent += self.prob_listener_referent_signal(order, referent_index, signal_index_set)

            return self.prob_listener_referent_signal(order, referent_index, signal_index)/sum_prob_referent

    def prob_listener_referent_signal(self, order, referent_index, signal_index):
        """
        Calculate the probability of the referent given the signal (reasoning about the listener): probability of the
        signal given the referent (speaker n-1) divided by the sum of the probabilities of the signal given all
        referents.
        :param order: int; the order of pragmatic reasoning of the speaker
        :param referent_index: int; the index of the referent in the lexicon
        :param signal_index: int; the index of the signal in the lexicon
        :return: float; the probability of the referent given the signal
        """

        sum_prob_signal = 0.0
        for referent_index_set in range(np.size(self.lexicon, 1)):
            sum_prob_signal += self.prob_signal_referent(order - 1, referent_index_set, signal_index)

        return self.prob_signal_referent(order - 1, referent_index, signal_index) / sum_prob_signal

# /////////////////////////////////////////////////// Interpretation ///////////////////////////////////////////////////
class Interpretation:

    def __init__(self, lexicon, signal, order, entropy_threshold, dialogue_history, order_threshold = 2):
        """
        Initialization of class.
        :param lexicon: array; the lexicon used by the listener
        :param signal: int; the received signal (index) from the speaker
        :param order: int; order of pragmatic reasoning of the listener
        :param entropy_threshold: int; the entropy threshold
        :param order_threshold: int; the order of pragmatic reasoning threshold
        :param dialogue_history: list; the previously produced signals by the speaker
        """

        self.lexicon = lexicon
        self.signal = signal
        self.order_threshold = order_threshold
        self.order = order
        self.dialogue_history = dialogue_history
        self.entropy_threshold = entropy_threshold

        self.norm_lex = self.normalise_lexicon()
        self.max_ent = 0

    def interpret(self):
        """
        Start interpreting the signal of the speaker by calling the function corresponding to the order of pragmatic
        reasoning of the listener.
        :return: inferred referent (int) or OIR (string) and whether the threshold of the order or pragmatic reasoning
        was reached (Boolean) by calling the corresponding production function
        """

        if self.order == 0:
            output, order_threshold_reached = self.interpretation_literal()
            return output, order_threshold_reached, self.max_ent
        else:
            output, order_threshold_reached = self.interpretation_pragmatic()
            return output, order_threshold_reached, self.max_ent

    def interpretation_literal(self):
        """
        Interpret the signal by calculating the posterior distribution of the referents given the signal, lexicon and
        dialogue history if not empty. The entropy over the posterior distribution decides how certain the listener is
        of the inferred referent: if uncertain, the listener will signal other-initiated repair (OIR) to give the turn
        to the speaker again. If certain, the listener will output the inferred referent.
        :return: output which can consist of either an inferred referent or other-initiated repair (OIR) and 0
        (meaning that the threshold of the order of pragmatic reasoning was not reached)
        """

        #Perform conjunction if a dialogue history is available, returning a combined signal (from previous signal +
        # current signal)
        if self.dialogue_history:
            self.lexicon[self.signal] = self.conjunction()
            self.norm_lex = self.normalise_lexicon()

        #Calculate posterior distribution given the signal, lexicon, and dialogue history if not empty
        max_referent = np.amax(self.norm_lex, axis=1)[self.signal]
        indices = np.where(self.norm_lex[self.signal] == max_referent)
        referent = int(np.random.choice(indices[0], 1))

        #Calculate conditional entropy of posterior distribution
        entropy = self.conditional_entropy()

        # when H < entropy_threshold: output inferred referent
        # output = referent --> call function
        if entropy <= self.entropy_threshold:
            output = referent

        # when H > entropy_threshold:
        # turn to speaker --> output = OIR
        if entropy > self.entropy_threshold:
            output = "OIR"

        return output, 0

    def interpretation_pragmatic(self):
        """
        Interpret the signal by calculating the posterior distribution of the referents given the signal, lexicon and
        dialogue history if not empty. The entropy over the posterior distribution decides how certain the listener is
        of the inferred referent: if uncertain, the listener will go a level up on the order of pragmatic reasoning and
        interpret the signal again, with a higher order of pragmatic reasoning. If certain, the listener will output
        the inferred referent.
        :return: inferred referent (int) and whether the threshold of the order of pragmatic reasoning is reached
        (Boolean)
        """

        #Initialize the variable for whether the threshold of the order of pragmatic reasoning is reached
        order_threshold_reached = 0

        #Calculate posterior distribution given signal and lexicon

        #Calculate the conditional entropy of posterior distribution
        entropy = self.conditional_entropy()

        #if H > entropy_threshold & order < order_threshold
        if (entropy > self.entropy_threshold) and (self.order < self.order_threshold):
            self.order += 1
            self.interpretation_pragmatic()

        # if H < entropy_threshold or when order = order_threshold --> recursion till order=0
        #save when order_threshold is reached!
        #interpretation(dialogue_history, lexicon, order, signal) --> call function!
        if (entropy <= self.entropy_threshold) or (self.order == self.order_threshold):
            prob_lex = np.zeros(np.size(self.lexicon, 1))
            for referent_index in range(np.size(self.lexicon, 1)):
                prob_lex[referent_index] = self.prob_referent_signal(self.order, referent_index, self.signal)

            max_prob_referent = np.amax(prob_lex)
            indices = np.where(prob_lex == max_prob_referent)
            referent = int(np.random.choice(indices[0], 1))

            if self.order == self.order_threshold:
                order_threshold_reached = 1

        return referent, order_threshold_reached

    def conditional_entropy(self):
        """
        Calculate the conditional entropy over the posterior distribution of the referents given the signal, lexicon and
        dialogue history if not empty.
        :return: int; the entropy of the posterior distribution
        """

        #Take the sum of: the probability of the referent given the signal and the lexicon times the log of 1 divided by
        #the probability of the referent given the signal and the lexicon
        sum_ref_prob = 0.0
        for referent_index in range(np.size(self.lexicon, 1)):
            prob_max_ent = 1/(np.size(self.lexicon, 1))
            prob = self.prob_referent_signal(self.order, referent_index, self.signal)
            self.max_ent += prob_max_ent * math.log((1 / prob_max_ent), 2)
            if prob != 0:
                sum_ref_prob += prob * math.log((1/prob),2)

        return sum_ref_prob

    def conjunction(self):
        """
        Perform conjunction between the current signal and the last produced signal by the speaker.
        :return: array; the conjunction of both signals into a combined signal
        """

        #Initialize an array to store the combined signal in
        combined_signal = np.zeros(self.lexicon.shape[1])

        #Perform the conjunction between the current and previous signal (from the dialogue history)
        index = 0

        for ref, ref2 in zip(self.lexicon[self.dialogue_history[-1]], self.lexicon[self.signal]):
            if ref == 1 and ref2 ==1:
                combined_signal[index] = 1
            index += 1

        return combined_signal

    def normalise_lexicon(self):
        """
        Calculate the probabilities of the referents given the signal to create a lexicon filled with probabilities.
        :return: array; lexicon with probabilities of referents given the signal
        """

        prob_lex_interpretation = np.transpose(np.divide(np.transpose(self.lexicon), np.sum(self.lexicon, axis=1),
                                               out=np.zeros_like(np.transpose(self.lexicon)), where=np.sum(self.lexicon, axis=1)!=0))

        return prob_lex_interpretation

    def prob_referent_signal(self, order, referent_index, signal_index):
        """
        Calculate the probability of the referent given the signal: probability of the signal given r (speaker: same
        order of pragmatic reasoning) divided by the sum of the (speaker) probabilities of the signal given all
        referents. If n = 0, calculate the literal probability.
        :param order: int; order of pragmatic reasoning of the listener
        :param referent_index: int; the index of the referent in the lexicon
        :param signal_index: int; the index of the signal in the lexicon
        :return: float; the probability of the referent given the signal.
        """

        #When the order is 0, the literal probability is calculated
        if order == 0:
            return self.norm_lex[signal_index][referent_index]
        #When the order is not 0, the pragmatic probabilities are calculated
        else:
            sum_prob_signal = 0.0
            for referent_index_set in range(np.size(self.lexicon, 1)):
                sum_prob_signal += self.prob_speaker_signal_referent(order, referent_index_set, signal_index)


            return self.prob_speaker_signal_referent(order, referent_index, signal_index)/sum_prob_signal

    def prob_speaker_signal_referent(self, order, referent_index, signal_index):
        """
        Calculate the probability of the signal given the referent (reasoning about the speaker): probability of the
        referent given the signal (listener n-1) divided by the sum of the probabilities of the referent given all
        signals.
        :param order: int; the order of pragmatic reasoning of the listener
        :param referent_index: int; the index of the referent in the lexicon
        :param signal_index: int; the index of the signal in the lexicon
        :return: float; the probability of the signal given the referent.
        """

        sum_prob_referent = 0.0
        for signal_index_set in range(np.size(self.lexicon, 0)):
            sum_prob_referent += self.prob_referent_signal(order - 1, referent_index, signal_index_set)

        return self.prob_referent_signal(order - 1, referent_index, signal_index) / sum_prob_referent


# ------------------------------------------------- Part 2: Simulation -------------------------------------------------
#                                            Simulate a Single Conversation

#Initializing Agents: order of pragmatic reasoning, ambiguity level lexicon, type (listener, speaker), optional: entropy threshold
class Agent:

    def __init__(self, order, agent_type, entropy_threshold):
        """
        Initialization of class.
        :param order: int; order of pragmatic reasoning of agent
        :param agent_type: string; speaker or listener type
        :param entropy_threshold: int; entropy threshold
        """
        self.order = order
        self.type = agent_type
        self.entropy_threshold = entropy_threshold

def interaction(speaker, listener, lexicon):
    """
    Perform one interaction (until the listener is certain enough about an inferred referent or the threshold for the
    order of pragmatic reasoning is reached) between listener and speaker.
    :param speaker: the agent with type "speaker"
    :param listener: the agent with type "listener"
    :param lexicon: array; lexicon for the speaker and listener (assumption: both agents have the same lexicon)
    :return: array; the array contains the following information about the interaction: the intention, the inferred
    referent by the listener, the amount of turns, the order of speaker and listener (assumed to be equal), the
    communicative success and whether the threshold of the order of pragmatic reasoning was reached
    """
    #Initialize the amount of turns and the dialogue history
    turns = 0
    dialogue_history = []

    #Generate intention: randomly generated from uniform distribution
    n_referents = lexicon.shape[1]
    intention = np.random.randint(n_referents)

    #Start interaction by the speaker producing a signal and the listener interpreting that signal, if (and as long as)
    #the listener signals other-initiated repair (OIR), the speaker and listener will continue the interaction by
    #producing and interpreting new signals.
    produced_signal = Production(lexicon, intention, speaker.order, dialogue_history).produce()
    turns += 1
    listener_output, order_threshold_reached, max_entropy = Interpretation(lexicon, produced_signal, listener.order, listener.entropy_threshold, dialogue_history, order_threshold=3).interpret()
    dialogue_history.append(produced_signal)
    turns += 1
    while listener_output == "OIR":
        produced_signal = Production(lexicon, intention, speaker.order, dialogue_history).produce()
        turns += 1
        listener_output, order_threshold_reached, max_entropy = Interpretation(lexicon, produced_signal, listener.order, listener.entropy_threshold, dialogue_history).interpret()
        dialogue_history.append(produced_signal)
        turns += 1
        if turns > 100:
            break

    #Save the wanted information in an array to be returned
    #QUESTION: DO WE WANT TO SAVE THE IN BETWEEN SIGNALS (all the produced signals in a conversation)? --> yes
    output = np.array([intention, listener_output, turns, speaker.order, communicative_success(intention, listener_output),
                       order_threshold_reached])

    return output, max_entropy

# ///////////////////////////////////////// Measurements: dependent variables /////////////////////////////////////////
def communicative_success(intention, referent):
    """
    Calculate the communicative success: 1 if the intention and inferred referent are equal, 0 otherwise.
    :param intention: int; the intention of the speaker
    :param referent: int; the inferred referent by the listener
    :return: int; communicative success
    """

    if intention == referent:
        com_suc = 1
    else:
        com_suc = 0

    return com_suc

def average_com_suc(interactions):
    """
    Calculate the average communicative success: average of communicative success over all interactions.
    :param interactions: array; every row represents an interaction with the intention of the speaker and the inferred
    referent from the listener
    :return: float; average communicative success
    """

    sum = 0
    for intention, referent in interactions:
        sum += communicative_success(intention, referent)

    return sum / len(interactions)

#Complexity: also a measurement, but not included here

# //////////////////////////////////////////////// Running Simulations ////////////////////////////////////////////////
#Think about multiprocessing
def simulation(n_interactions, ambiguity_level, n_signals, n_referents, order, entropy_threshold):
    """
    Run a simulation of a number of interactions (n_interactions), with the specified parameters.
    :param n_interactions: int; the number of interactions to be performed in the simulation
    :param ambiguity_level: float (between 0.0 and 1.0); the desired ambiguity level of the lexicon
    :param n_signals: int; the number of signals in the lexicon
    :param n_referents: int; the number of referents in the lexicon
    :param order: int; the order of pragmatic reasoning for both agents
    :param entropy_threshold: int; the entropy threshold
    :return: dataframe; consisting of the following information: the intention of the speaker, the inferred referent of
    the listener, the number of turns, the order of pragmatic reasoning, the communicative success, whether the
    threshold of the order of pragmatic reasoning was reached, the ambiguity level, the number of signals, the number
    of referents
    """
    #Initialize agents with the order of pragmatic reasoning, agent type, and entropy threshold
    speaker = Agent(order, "Speaker", entropy_threshold)
    listener = Agent(order, "Listener", entropy_threshold)

    #Generate Lexicons with the number of signals, the number of referents, the ambiguity level and the number of
    #lexicons
    lexicons_df = pd.read_json('lexiconset.json')
    n_lexicons = n_interactions
    lexicons = lex_retriever.retrieve_lex(lexicons_df, n_signals, n_referents, ambiguity_level, n_lexicons)

    #Initliaze pandas dataframe to store results
    results = pd.DataFrame(
        columns=["Intention Speaker", "Inferred Referent Listener", "Number of Turns", "Order of Reasoning",
                 "Communicative Success", "Reached Threshold Order", "Ambiguity Level", "Number of Signals",
                 "Number of referents"])

    #Run the desired number of interactions for the simulation and store the results in the pandas dataframe
    for i in range(n_interactions):
        result, max_entropy = interaction(speaker, listener, lexicons[i])
        results.loc[len(results)] = np.concatenate((result, np.array([ambiguity_level, n_signals, n_referents])), axis=None)

    print("Maximum entropy: ", max_entropy)
    print(results)
    return results

simulation(10, 0.5, 10, 8, 0, 1)