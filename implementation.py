# Necessary imports
import pandas as pd
import numpy as np
import lexicon_retriever as lex_retriever
import math
import pickle
import multiprocessing

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

        # Initialize a variable norm_lex to store the normalised lexicon in
        self.norm_lex = self.normalise_lexicon_for_production()

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

        # Perform conjunction if a dialogue history is available: the lexicon changes accordingly for this interaction
        if self.dialogue_history:
            self.lexicon = self.conjunction_for_production()
            self.norm_lex = self.normalise_lexicon_for_production()

        # Calculate which signal maximizes the probability, given the intention
        signal = self.pick_max_signal(self.norm_lex)

        return signal

    def production_pragmatic(self):
        """
        For pragmatic speakers, a signal is produced by choosing the signal that maximizes the probability given the
        intention, determined by the order of pragmatic reasoning.
        :return: int; signal
        """

        # Calculate which signal maximizes the probability
        probs_production = self.norm_lex
        for _ in range(self.order):
            probs_reasoning_interpretation = np.divide(probs_production.T, np.sum(probs_production, axis=1)).T
            probs_production = np.divide(probs_reasoning_interpretation, np.sum(probs_reasoning_interpretation, axis=0))

        signal = self.pick_max_signal(probs_production)

        return signal

    def conjunction_for_production(self):
        """
        Perform conjunction between the lexicon and the last produced signal.
        :return: array; new lexicon based on the conjunction
        """

        # Perform conjunction on last produced signal and the lexicon
        new_lexicon = np.multiply(self.lexicon, self.lexicon[self.dialogue_history[-1]])

        return new_lexicon

    def normalise_lexicon_for_production(self):
        """
        Calculate the probabilities of the signals given the referents to create a lexicon filled with probabilities.
        :return: array; lexicon with probabilities of signals given referents
        """

        prob_lex_production = np.divide(self.lexicon, np.sum(self.lexicon, axis=0), out=np.zeros_like(self.lexicon),
                                        where=np.sum(self.lexicon, axis=0) != 0)

        return prob_lex_production

    def pick_max_signal(self, prob_lex):
        """
        Pick the signal given the intention with the highest probability.
        :param prob_lex: array; the probability lexicon of the speaker, dependent on the order of pragmatic reasoning
        :return: int; the signal with the highest probability given the referent (intention)
        """
        max_signal = np.amax(prob_lex.T[self.intention])
        indices = np.where(prob_lex.T[self.intention] == max_signal)
        signal = int(np.random.choice(indices[0], 1))

        return signal

# /////////////////////////////////////////////////// Interpretation ///////////////////////////////////////////////////
class Interpretation:

    def __init__(self, lexicon, signal, order, entropy_threshold, dialogue_history, order_threshold=2):
        """
        Initialization of class.
        :param lexicon: array; the lexicon used by the listener
        :param signal: int; the received signal (index) from the speaker
        :param order: int; order of pragmatic reasoning of the listener
        :param entropy_threshold: float; the entropy threshold
        :param order_threshold: int; the order of pragmatic reasoning threshold
        :param dialogue_history: list; the previously produced signals by the speaker
        """

        self.lexicon = lexicon
        self.signal = signal
        self.order_threshold = order_threshold
        self.order = order
        self.dialogue_history = dialogue_history
        self.entropy_threshold = entropy_threshold

        # Initialize a variable norm_lex to store the normalised lexicon in
        self.norm_lex = self.normalise_lexicon_for_interpretation()

    def interpret(self):
        """
        Start interpreting the signal of the speaker by calling the function corresponding to the order of pragmatic
        reasoning of the listener.
        :return: inferred referent (int) or OIR (string) and whether the threshold of the order or pragmatic reasoning
        was reached (Boolean) by calling the corresponding interpretation function
        """

        if self.order == 0:
            output, order_threshold_reached = self.interpretation_literal()
            return output, order_threshold_reached
        else:
            output, order_threshold_reached = self.interpretation_pragmatic()
            return output, order_threshold_reached

    def interpretation_literal(self):
        """
        Interpret the signal by calculating the posterior distribution of the referents given the signal, lexicon and
        dialogue history if not empty. The entropy over the posterior distribution decides how certain the listener is
        of the inferred referent: if uncertain, the listener will signal other-initiated repair (OIR) to give the turn
        to the speaker again. If certain, the listener will output the inferred referent.
        :return: output which can consist of either an inferred referent or other-initiated repair (OIR) and False
        (meaning that the threshold of the order of pragmatic reasoning was not reached)
        """

        # Perform conjunction if a dialogue history is available, returning a combined signal (from previous signal +
        # current signal) and normalize the new lexicon to get a new posterior distribution according to the changes
        # due to the conjunction operation
        if self.dialogue_history:
            self.lexicon[self.signal] = self.conjunction_for_interpretation()
            self.norm_lex = self.normalise_lexicon_for_interpretation()

        # Calculate conditional entropy of posterior distribution
        entropy = self.conditional_entropy(self.norm_lex)

        # When the entropy <= entropy_threshold return the referent
        if entropy <= self.entropy_threshold:
            output = self.pick_max_referent(self.norm_lex)
        else:
        # When the entropy > entropy_threshold return an OIR signal
            output = 'OIR'

        return output, False

    def interpretation_pragmatic(self):
        """
        Interpret the signal by calculating the posterior distribution of the referents given the signal, lexicon and
        dialogue history if not empty. The entropy over the posterior distribution decides how certain the listener is
        of the inferred referent: if uncertain, the listener will go a level up on the order of pragmatic reasoning
        (until the order threshold is reached) and interpret the signal again, with a higher order of pragmatic
        reasoning. If certain, the listener will output the inferred referent.
        :return: inferred referent (int) and whether the threshold of the order of pragmatic reasoning is reached
        (Boolean)
        """

        # Initialize the variable for whether the threshold of the order of pragmatic reasoning is reached
        order_threshold_reached = False

        # Calculate the posterior distribution
        probs_interpretation = self.norm_lex
        for _ in range(self.order):
            probs_reasoning_production = np.divide(probs_interpretation, np.sum(probs_interpretation, axis=0))
            probs_interpretation = np.divide(probs_reasoning_production.T, np.sum(probs_reasoning_production, axis=1)).T

        # Calculate the conditional entropy of the posterior distribution
        entropy = self.conditional_entropy(probs_interpretation)

        # If the entropy > entropy_threshold and the order of pragmatic reasoning is smaller than its threshold
        # interpret the signal again with one level up on pragmatic reasoning
        if (entropy > self.entropy_threshold) and (self.order < self.order_threshold):
            self.order += 1
            self.interpretation_pragmatic()

        # If the entropy <= than the entropy threshold or when the order of pragmatic reasoning is equal to its
        # threshold, the referent with the highest probability given the signal is calculated and outputted
        if entropy <= self.entropy_threshold:
            referent = self.pick_max_referent(probs_interpretation)
        elif self.order == self.order_threshold:
            referent = self.pick_max_referent(probs_interpretation)
            order_threshold_reached = True

        return referent, order_threshold_reached

    def conditional_entropy(self, probs_interpretation):
        """
        Calculate the conditional entropy over the posterior distribution of the referents given the signal, lexicon and
        dialogue history if not empty.
        :param interpretation_probs: array; the posterior distribution as the probability lexicon
        :return: float; the entropy of the posterior distribution
        """

        # Take the sum of: the probability of the referent given the signal and the lexicon times the log of 1 divided
        # by the probability of the referent given the signal and the lexicon

        conditional_entropy = 0.0
        for referent_index in range(np.size(self.lexicon, 1)):
            prob = probs_interpretation[self.signal][referent_index]
            if prob != 0:
                conditional_entropy += prob * math.log((1 / prob), 2)

        return conditional_entropy

    def conjunction_for_interpretation(self):
        """
        Perform conjunction between the current signal and the last produced signal by the speaker.
        :return: array; the conjunction of both signals into a combined signal
        """

        # Perform the conjunction between the current and previous signal (from the dialogue history)
        combined_signal = np.multiply(self.lexicon[self.signal], self.lexicon[self.dialogue_history[-1]])

        return combined_signal

    def normalise_lexicon_for_interpretation(self):
        """
        Calculate the probabilities of the referents given the signal to create a lexicon filled with probabilities.
        :return: array; lexicon with probabilities of referents given the signal
        """

        prob_lex_interpretation = np.transpose(np.divide(np.transpose(self.lexicon), np.sum(self.lexicon, axis=1),
                                                         out=np.zeros_like(np.transpose(self.lexicon)),
                                                         where=np.sum(self.lexicon, axis=1) != 0))

        return prob_lex_interpretation

    def pick_max_referent(self, prob_lex):
        """
        Pick the referent given the received signal with the highest probability.
        :param prob_lex: array; the probability lexicon of the listener, dependent on the order of pragmatic reasoning
        :return: int; the referent with the highest probability given the received signal
        """
        max_prob_referent = np.amax(prob_lex[self.signal])
        indices = np.where(prob_lex[self.signal] == max_prob_referent)
        referent = int(np.random.choice(indices[0], 1))

        return referent

# ------------------------------------------------- Part 2: Simulation -------------------------------------------------
#                                            Simulate a Single Conversation

# Initializing Agents: order of pragmatic reasoning, type (listener, speaker)
class Agent:

    def __init__(self, order, agent_type, entropy_threshold):
        """
        Initialization of class.
        :param order: int; order of pragmatic reasoning of agent
        :param agent_type: string; speaker or listener type
        :param entropy_threshold: float; the entropy threshold of the agent, in order to decide when the agent is
        certain/uncertain when it comes to entropy.
        """
        self.order = order
        self.type = agent_type
        self.entropy_threshold = entropy_threshold


def interaction(speaker, listener, lexicon):
    """
    Perform one interaction (until the listener is certain enough about an inferred referent or the threshold for the
    order of pragmatic reasoning is reached) between listener and speaker.
    :param speaker: agent; the agent with type "speaker"
    :param listener: agent; the agent with type "listener"
    :param lexicon: array; lexicon for the speaker and listener (assumption: both agents have the same lexicon)
    :return: array; the array contains the following information about the interaction: the intention, the inferred
    referent by the listener, the amount of turns, the order of speaker and listener, the communicative success and
    whether the threshold of the order of pragmatic reasoning was reached
    """
    # Initialize the amount of turns, whether the interaction threshold is reached, the dialogue history and the state
    # of the listener
    turns = 0
    interaction_threshold_reached = False
    dialogue_history = []
    listener_state = 'start'

    # Generate intention: randomly generated from uniform distribution
    n_referents = lexicon.shape[1]
    intention = np.random.randint(n_referents)

    # Start interaction by the speaker producing a signal and the listener interpreting that signal, if (and as long as)
    # the listener signals other-initiated repair (OIR), the speaker and listener will continue the interaction by
    # producing and interpreting new signals
    while listener_state == 'OIR' or listener_state == 'start':
        produced_signal = Production(lexicon, intention, speaker.order, dialogue_history).produce()
        turns += 1
        listener_output, order_threshold_reached = Interpretation(lexicon, produced_signal, listener.order,
                                                                               listener.entropy_threshold,
                                                                  dialogue_history, order_threshold=2).interpret()
        listener_state = listener_output
        dialogue_history.append(produced_signal)
        turns += 1

        # To avoid infinite loops, when you reach a number of turns that is bigger than the twice the number of signals,
        # the interaction is stopped
        if turns >= (2 * lexicon.shape[0]):
            interaction_threshold_reached = True
            break

    # Calculate the communicative success of the interaction
    success = communicative_success(intention, listener_output)

    # Save the wanted information in a pandas dataframe to be returned
    output = pd.DataFrame([intention, listener_output, turns, speaker.order, listener.order, success,
                                   order_threshold_reached, interaction_threshold_reached, dialogue_history],
                          columns=['Intention Speaker', 'Inferred Referent Listener', 'Number of Turns',
                                   'Order of Reasoning Speaker', 'Order of Reasoning Listener', 'Communicative Success',
                                   'Reached Threshold Order', 'Reached Threshold Interaction', 'Dialogue History'])
    print(output)
    return output


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

# Complexity: also a measurement, but not included here

# //////////////////////////////////////////////// Running Simulations ////////////////////////////////////////////////
def simulation(n_interactions, ambiguity_level, n_signals, n_referents, speaker_order, listener_order, entropy_threshold,
               n_runs_simulation):
    """
    Run a simulation of a number of interactions (n_interactions), with the specified parameters.
    :param n_interactions: int; the number of interactions to be performed in the simulation
    :param ambiguity_level: float (between 0.0 and 1.0); the desired ambiguity level of the lexicon
    :param n_signals: int; the number of signals in the lexicon
    :param n_referents: int; the number of referents in the lexicon
    :param speaker_order: int; the order of pragmatic reasoning for the speaker
    :param listener_order: int; the order of pragmatic reasoning for the listener
    :param entropy_threshold: int; the entropy threshold
    :param n_runs_simulation: int; the number of runs of the simulation
    :return: dataframe; consisting of the following information: the intention of the speaker, the inferred referent of
    the listener, the number of turns, the order of pragmatic reasoning, the communicative success, whether the
    threshold of the order of pragmatic reasoning was reached, the ambiguity level, the number of signals, the number
    of referents
    """

    # Initialize agents with the order of pragmatic reasoning, agent type, and entropy threshold
    speaker = Agent(speaker_order, 'Speaker', entropy_threshold)
    listener = Agent(listener_order, 'Listener', entropy_threshold)

    # Generate Lexicons with the number of signals, the number of referents, the ambiguity level and the number of
    # lexicons
    lexicons_df = pd.read_json('lexiconset3.json')
    n_lexicons = n_runs_simulation
    lexicons = lex_retriever.retrieve_lex(lexicons_df, n_signals, n_referents, ambiguity_level, n_lexicons)

    # Initialize pandas dataframe to store results
    results = pd.DataFrame(
        columns=['Intention Speaker', 'Inferred Referent Listener', 'Number of Turns', 'Order of Reasoning Speaker',
                 'Order of Reasoning Listener', 'Communicative Success', 'Reached Threshold Order',
                 'Reached Threshold Interaction', 'Dialogue History', 'Ambiguity Level', 'Number of Signals',
                 'Number of Referents', 'Entropy Threshold'])

    general_info = pd.DataFrame([ambiguity_level, n_signals, n_referents, entropy_threshold])

    pool = multiprocessing.Pool(processes=16)
    for _ in range(n_interactions):
        arguments = zip([speaker]*n_lexicons,[listener]*n_lexicons,lexicons)
        arg_list = list(arguments)
        result = pool.starmap(interaction, arg_list)
        results.loc[len(results)] = pd.concat([result[0], general_info], axis=1)

    # Make sure that the values are integers in order to take the mean
    results['Reached Threshold Order'] = results['Reached Threshold Order'].astype(int)
    results['Reached Threshold Interaction'] = results['Reached Threshold Interaction'].astype(int)
    results['Communicative Success'] = results['Communicative Success'].astype(int)
    results['Number of Turns'] = results['Number of Turns'].astype(int)

    # Take the mean of the specified variables
    simulation_averages = {'Reached Threshold Order': results['Reached Threshold Order'].mean(),
                           'Reached Threshold Interaction': results['Reached Threshold Interaction'].mean(),
                           'Communicative Success': results['Communicative Success'].mean(),
                           'Number of Turns': results['Number of Turns'].mean()}

    # Pickle the results
    filename = 'amb_' + float_to_string(ambiguity_level) + '_lex_' + str(n_signals) + 'x' + str(n_referents) + '_orderS_' +\
               str(order_speaker) + '_orderL_' + str(order_listener) + '_ent_' + float_to_string(entropy_threshold) + \
               '_nInter_' + str(n_interactions) + '_nSim_' + str(n_runs_simulation) + '.p'
    outfile = open(filename, 'wb')
    pickle.dump(results, outfile)
    outfile.close()

    # print(simulation_averages)

def float_to_string(float_object):
    """
    Takes a float and returns it as a string where the '.' is replaced by a ','.
    :param float_object: float; the float to be converted
    :return: string; the float converted into a string where the '.' is replaced by a ','
    """
    string = str(float_object)
    string_new = string.replace('.', ',')

    return string_new
