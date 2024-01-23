import time

import matplotlib.pyplot as plt
import mobspy.modules.meta_class_utils as mcu
import numpy as np
from mobspy.modules.function_rate_code import *
import reaction_construction as rc
import mobspy.modules.function_rate_code as frc
from pint import Quantity
from mobspy.modules.order_operators import *
import mobspy.modules.event_functions as eh
from mobspy.modules.logic_operator_objects import *
import mobspy.modules.species_string_generator as ssg
from copy import deepcopy
from mobspy.modules.mobspy_parameters import *
from mobspy.modules.mobspy_expressions import *
import itertools
from joblib import Parallel, delayed
import random
import joblib
import re
import scipy.linalg as sclig
import bound_animator as ba
from numba import njit, float64


def q_compile(simulation_object):
    # Check to see if all species are named
    # Parameter compilation as well

    # Start by extracting the necessary things from the simulation object
    meta_species_to_simulate = simulation_object.model
    volume = simulation_object.parameters['volume']
    orthogonal_vector_structure = simulation_object.orthogonal_vector_structure
    reactions_set = simulation_object._reactions_set

    names_used = set()

    # Removing repeated elements to ensure only one meta-species in the simulation
    meta_species_to_simulate = meta_species_to_simulate.remove_repeated_elements()
    black_listed_names = {'Time', 'Rev', 'All'}
    for i, species in enumerate(meta_species_to_simulate):
        if '_dot_' in species.get_name():
            simlog.error(f'In species: {species.get_name()} \n _dot_ cannot be used in meta-species names')
        if species.get_name() in black_listed_names:
            simlog.error(f'The name {species.get_name()} is not allowed for meta-species please change it')
        if '$' in species.get_name():
            simlog.error(f'In species: {species.get_name()} \n'
                         f'An error has occurred and one of the species was either not named or named with the '
                         f'restricted $ symbol')
        if species.get_name() in names_used:
            simlog.error(f'Names must be unique for all species\n' +
                         f'The repeated name is {species.get_name()} in position {i}\n' +
                         f'Another possibility could be a repeated meta-species in the model')
        names_used.add(species.get_name())

    # List of Species objects
    species_list = {}
    mappings = {}
    for spe_object in meta_species_to_simulate:
        species_string_list = ssg.construct_all_combinations(spe_object, 'std$',
                                                             orthogonal_vector_structure)
        for x in species_string_list:
            x[0] = x[0].get_name()
        mappings[spe_object.get_name()] = ['.'.join(x) for x in species_string_list]
        for species_string in species_string_list:
            species_list['.'.join(species_string)] = 0
    set_species_model = set(species_list.keys())

    species_counts = []
    for spe_object in meta_species_to_simulate:
        for count in spe_object.get_quantities():
            species_counts.append({'object': spe_object, 'characteristics': count['characteristics'],
                                   'quantity': count['quantity']})

    initial_counts = {}

    # Add initial state counts
    for count in species_counts:

        if 'all$' not in count['characteristics']:
            continue

        temp_set = set(count['characteristics'])
        temp_set.remove('all$')
        species_strings = ssg.construct_all_combinations(count['object'], temp_set,
                                                         orthogonal_vector_structure,
                                                         symbol='_dot_')

        temp_count = uh.convert_counts(count['quantity'], volume, 3)
        for spe_str in species_strings:
            initial_counts[spe_str] = int(temp_count)

    # Set initial counts for SBML
    # Create the list HERE
    for count in species_counts:

        if 'all$' in count['characteristics']:
            continue

        species_string = ssg.construct_species_char_list(count['object'], count['characteristics'],
                                                         orthogonal_vector_structure,
                                                         symbol='_dot_')

        temp_count = uh.convert_counts(count['quantity'], volume, 3)
        initial_counts[species_string] = int(temp_count)

    # Default dimension equal to three after update
    dimension = None
    if isinstance(volume, Quantity):
        dimension = uh.extract_length_dimension(str(volume.dimensionality), dimension)
    else:
        dimension = 3

    # Check volume:
    volume = uh.convert_volume(volume, dimension)

    # BaseSpecies reactions for SBML with theirs respective parameters and rates
    # What do I have so far
    # Species_String_Dict and a set of reaction objects in Reactions_Set
    mobspy_parameters = set()
    reactions_for_q = rc.create_all_reactions(reactions_set,
                                              meta_species_to_simulate,
                                              orthogonal_vector_structure,
                                              dimension,
                                              mobspy_parameters)

    parameters_for_sbml = {'volume': (volume, f'dimensionless')}
    # cls.add_to_parameters_to_sbml(parameters_used, parameters_for_sbml, parameters_in_reaction)

    # We do not check for duplicate reactions in the q_compilation

    # Check to see if parameters are names are repeated or used as meta-species
    for p in parameters_for_sbml:
        if p in names_used:
            simlog.error('Parameters names must be unique and they must not share a name with a species')
        names_used.add(p)

    # No MobsPy parameters accepted for now - I should most definitely correct this
    return set_species_model, initial_counts, reactions_for_q


def q_matrix_generator(set_species_model, reactions_for_q, interval, exit_direction):
    # Transform user format to proper format used for matrix generation

    interval = process_initial_counts_dict(interval)

    for spe in set_species_model:
        if spe not in interval.keys():
            simlog.error("Species: " + str(spe) + "\n"
                                                  "There is a species on the simulation whose interval was not specified")

    all_value_combinations = []
    species_order_index = {}
    for i, spe in enumerate(sorted(interval.keys())):
        species_order_index[spe] = i
        if interval[spe][0] <= interval[spe][1]:
            l_v = list(range(interval[spe][0], interval[spe][1] + 1))
        else:
            l_v = list(range(interval[spe][1], interval[spe][0] + 1))
            interval[spe] = [interval[spe][1], interval[spe][0]]
        all_value_combinations.append(l_v)

    if exit_direction is not None:
        rev_exit_direction = process_exit_dictionary(exit_direction, species_order_index, interval)
    else:
        rev_exit_direction = {}

    # Index for matrix creation
    state_index_to_spe, spe_to_state_index = {}, {}
    for i, comb in enumerate(itertools.product(*all_value_combinations)):
        state_index_to_spe[comb], spe_to_state_index[i] = i, comb

    row_gen = lambda i, state: q_row_generation(state, reactions_for_q, state_index_to_spe, species_order_index,
                                                rev_exit_direction)
    q_matrix = Parallel(n_jobs=-1, prefer="threads")(delayed(row_gen)(state_index_to_spe[state], state)
                                                     for state in state_index_to_spe.keys())
    q_matrix.append([0 for _ in range(len(state_index_to_spe) + 1)])

    q_matrix = np.array(q_matrix)

    for i, row in enumerate(q_matrix):
        q_matrix[i, i] = - sum(row)

    return q_matrix, state_index_to_spe, species_order_index


def q_row_generation(state, reactions_for_q, state_index_to_spe, species_order_index, rev_exit_direction):
    q_row = np.zeros(len(state_index_to_spe) + 1)

    for _, reaction in reactions_for_q.items():

        index_of_species_values_in_state = []
        for reactant in reaction['reactants']:
            index_of_species_values_in_state.append(species_order_index[reactant])

        state_for_reaction = [state[i] for i in index_of_species_values_in_state]
        state_for_reaction = tuple(state_for_reaction[0: reaction['argument_number']])

        q_value = reaction['rate_function'](state_for_reaction)

        new_state = list(state)
        for spe, delta in reaction['delta_species'].items():
            i = species_order_index[spe]
            new_state[i] = state[i] + delta
        new_state = tuple(new_state)

        # If the rate is inside interval, we add it to the state transition
        try:
            j = state_index_to_spe[new_state]
            q_row[j] = q_value
        # If not, we add it to the final absorbing state
        except KeyError:
            flag = False
            for key in rev_exit_direction:
                new_value = new_state[key]
                if rev_exit_direction[key][0] == '+' and new_value > rev_exit_direction[key][1]:
                    flag = True
                    break
                elif rev_exit_direction[key][0] == '-' and new_value < rev_exit_direction[key][1]:
                    flag = True
                    break
            if not flag:
                q_row[-1] = q_row[-1] + q_value

    return q_row


def process_initial_counts_dict(initial_counts):
    # Function that takes the user initial_counts dictionary and transforms in an output of the following format
    # {species: (initial_value, final_value) - } for all species in the simulation
    # For now it is incomplete
    new_dict = {}
    for key in initial_counts:
        new_dict[str(key)] = initial_counts[key]
    return new_dict


def process_exit_dictionary(exit_dictionary, species_order_index, interval):
    """
    :param species_order_index: species string name key = index in states value
    :param exit_dictionary: Function that takes the exit direction dictionary and transforms
    the keys according to each species index.
    :return: dictionary with species index as keys and allowed exit direction as values
    """
    initial_keys = list(exit_dictionary.keys())
    for key in initial_keys:
        exit_dictionary[str(key)] = exit_dictionary[key]
        del exit_dictionary[key]

    return_dict = {}
    for key in exit_dictionary:
        try:
            new_key = species_order_index[str(key)]
        except KeyError:
            continue
        if exit_dictionary[key] == 'bellow' or exit_dictionary[key] == '-':
            return_dict[new_key] = ('+', interval[key][1])
        elif exit_dictionary[key] == 'above' or exit_dictionary[key] == '+':
            return_dict[new_key] = ('-', interval[key][0])
        else:
            simlog.error(f'The string {exit_dictionary[key]} from key f{key}'
                         f'is not compatible with the direction of exit from the '
                         f'interval. Please check the available strings in the repository')

    return return_dict


def check_connectivity(q_matrix):
    available_states = set(range(0, len(q_matrix) - 1))

    def bdf(q):
        visitable_states = {0}
        state_stack = [0]
        initial_state = state_stack.pop(0)
        current_state = initial_state
        flag = True
        while flag:
            for i in range(len(q)):
                if q[current_state, i] > 0:
                    if i not in visitable_states:
                        state_stack.append(i)

                    visitable_states.add(i)

            try:
                current_state = state_stack.pop(0)
            except IndexError:
                flag = False

        return visitable_states

    forward_states, reversed_states = bdf(q_matrix), bdf(q_matrix.transpose())
    if bdf(q_matrix) == available_states and bdf(q_matrix.transpose()) == available_states:
        return True, 0, 0
    else:
        return False, forward_states, reversed_states


def convert_initial_counts_to_initial_state(initial_counts):
    initial_state = [initial_counts[key] for key in sorted(initial_counts.keys())]
    return tuple(initial_state)


def construct_jump_chain(q_matrix):
    # @TODO add one to last line of matrix
    J = np.zeros(q_matrix.shape)

    for i in range(q_matrix.shape[0]):
        if q_matrix[i, i] == 0:
            J[i, i] = 1
            continue

        for j in range(q_matrix.shape[0]):
            if i == j:
                continue
            J[i, j] = q_matrix[i, j] / -q_matrix[i, i]

        J[i, i] = 0

    return J


# @TODO Write here
def construct_cols_for_multiplication(q_matrix):

    non_zero_indexes = []
    for i in range(len(q_matrix)):
        col = []
        for j in range(len(q_matrix)):
            if q_matrix[j, i] > 0:
                col.append(j)
        non_zero_indexes.append(col)

    return non_zero_indexes


def bound_estimator(initial_counts, state_index_to_spe, q_matrix, species_order_index,
                    time_for_stationary_calculation=1e10,
                    max_iterations_per_diameter=1000, animation=False,
                    probability_epsilon_bound=1e-5,
                    stopping_probability=1e-5,
                    time_for_decay_estimation=1):
    initial_state = convert_initial_counts_to_initial_state(initial_counts)

    try:
        initial_state_index = state_index_to_spe[initial_state]
    except KeyError:
        simlog.error("The initial counts assignments were not found in the specified state bounds. "
                     "Thus, the initial distribution is not contained in the specified bounds")

    # The conditional q_matrix refers to the probability of not leaving the interval through the absorbed state
    # The absorbed state is represented by the last index in the matrix
    conditional_q_matrix = deepcopy(np.asarray(q_matrix))

    # We set the probability of reaching the absorbed state to zero
    conditional_q_matrix[:, -1] = 0

    # Recalculate the center to be the negative of the sum of all the new rates
    for i, row in enumerate(conditional_q_matrix):
        conditional_q_matrix[i, i] = 0
        conditional_q_matrix[i, i] = - sum(row)

    bol, forward_states, backward_states = check_connectivity(conditional_q_matrix)
    if bol:
        pass
    else:
        message = 'The states in the CRN are not connected. The bound does not apply \n'
        message += f'the state 0 can only reach: {forward_states} \n'
        message += f'the state 0 can only be reached by: {backward_states} \n'
        message += f'the states as species counts: \n'
        message += f'Species indexes: ' + str(species_order_index) + '\n'
        message += f'State numbers: \n'
        message += str(state_index_to_spe) + '\n'
        simlog.error(message)

    # qsd - quasy-stationary distribution
    qsd = np.zeros((len(q_matrix)))
    qsd[0] = 1

    t = time_for_stationary_calculation
    qsd = np.matmul(qsd, sclig.expm(t * conditional_q_matrix))
    q_array = deepcopy(qsd)

    # ind - initial distribution
    ind = np.zeros((len(q_matrix)))
    ind[initial_state_index] = 1

    # Construct the jump chain from the q_chain
    j_matrix = construct_jump_chain(q_matrix)
    index_for_mul_matrix = construct_cols_for_multiplication(q_matrix)

    # Animation of the bound calculation in effect
    if animation:
        ba.bound_animation(bound_array=ind, q_array=q_array, j_matrix=j_matrix)

    bound_array = np.matmul(ind, j_matrix)

    # the function bellow calculates one step of the bound
    @njit
    def process_bound_q(bound_array, q_array):
        for i in range(len(q_array)):
            bound_val, q_val = bound_array[i], q_array[i]

            if bound_val > 0:
                if bound_val - q_val <= 0:
                    q_array[i] = q_val - bound_val
                    bound_array[i] = 0
                else:
                    bound_array[i] = bound_val - q_val
                    q_array[i] = 0

    # Define my own matrix multiplication function to speed the process even further
    # It uses the fact that most indexes from matrix q are zero in CRNs
    @njit
    def faster_matrix_multiplication(bound_array, j_matrix, new_array, index_for_mul_matrix):
        for i in range(len(index_for_mul_matrix)):
            col = index_for_mul_matrix[i]
            soma = 0
            for index in col:
                soma = soma + bound_array[index]*j_matrix[index]
            new_array[i] = soma
        return new_array

    max_iterations = max_iterations_per_diameter * len(q_matrix)

    k, bound_sum = 0, 0
    a = time.time()
    while k < max_iterations:

        process_bound_q(bound_array, q_array)
        new_array = np.zeros(len(bound_array))
        bound_array = faster_matrix_multiplication(bound_array, j_matrix, new_array, faster_matrix_multiplication)

        k += 1
        if k % len(q_matrix) == 0:
            previous_bound_sum = bound_sum
            bound_sum = sum(bound_array)
            if abs(bound_sum - previous_bound_sum) * 100 <= probability_epsilon_bound:
                break
            if sum(bound_array) * 100 <= stopping_probability:
                break
    b = time.time()
    print(b - a)
    exit()

    print(sum(bound_array))
    print(abs(bound_sum - previous_bound_sum))
    print(k, sum(q_array))
    exit()

    bound_sum = sum(bound_array)

    # Here we estimate P(T > t) to calculate the decay parameter
    # With the decay parameter, we have the biggest non-zero eigenvalue
    # Prediction of qsds
    pe = 1 - np.matmul(qsd, sclig.expm(time_for_decay_estimation * q_matrix))[-1]
    decay_parameter = - np.log(pe) / time_for_decay_estimation

    return bound_sum, decay_parameter, qsd


class Jump_chain_qsd_bound:

    def __init__(self, simulation_object, interval, exit_direction=None):
        self.exit_direction = exit_direction
        self.simulation_object = simulation_object
        self.interval = interval
        self.species, self.counts, self.functional_rate_model = q_compile(simulation_object)

        # q_matrix, state_to_index = q_matrix_generator(species, resulting_comp, interval)
        self.q_matrix, self.state_to_index, self.species_order_index = q_matrix_generator(self.species,
                                                                                          self.functional_rate_model,
                                                                                          self.interval,
                                                                                          self.exit_direction)

        self.bound_constant, self.decay_parameter, self.qsd = None, None, None

    def calculate_bound(self, time_for_stationary_calculation=1e10, max_iterations_per_diameter=1000, animation=False,
                        probability_epsilon_bound=1e-5, stopping_probability=1e-5, time_for_decay_estimation=1):

        self.bound_constant, self.decay_parameter, \
        self.qsd = bound_estimator(self.counts, self.state_to_index,
                                   self.q_matrix,
                                   self.species_order_index,
                                   time_for_stationary_calculation=time_for_stationary_calculation,
                                   max_iterations_per_diameter=max_iterations_per_diameter,
                                   animation=False,
                                   probability_epsilon_bound=probability_epsilon_bound,
                                   stopping_probability=stopping_probability,
                                   time_for_decay_estimation=time_for_decay_estimation)

        print(self.bound_constant, self.decay_parameter)
        return self.bound_constant, self.decay_parameter

    def animate_bound(self, animation=True):

        self.bound_constant, self.decay_parameter, self.qsd = bound_estimator(self.counts, self.state_to_index,
                                                                              self.q_matrix, self.species_order_index,
                                                                              animation=True)

    def plot_qsd(self):
        if self.qsd is not None:
            plt.plot(self.qsd)
            plt.xlabel('State Index')
            plt.ylabel('Probability')
            plt.show()
        else:
            simlog.error('Qsd was not yet calculated. Please use method calculate bound to compute the qsd')
