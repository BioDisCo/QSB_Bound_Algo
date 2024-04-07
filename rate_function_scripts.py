import mobspy.modules.unit_handler as uh
import mobspy.modules.reaction_construction_nb as rc
from pint import Quantity
from mobspy.modules.mobspy_parameters import *
from called_argument_class import Species_Argument_For_Callable
import inspect


def mass_action_kinetics(reactant_string_list, reaction_rate):
    counts = rc.count_string_dictionary(reactant_string_list)

    def rate_function(counts, args):
        to_return = reaction_rate

        for arg, reactant in zip(args, reactant_string_list):
            i = counts[reactant] - 1

            to_return *= (arg - i) / (i + 1)

        return to_return

    return lambda r: rate_function(counts, r)


def transform_callable_rate(reacting_meta_species, reactant_string_list, reaction_rate):
    signature = inspect.signature(reaction_rate)
    argument_number = len(signature.parameters)

    def rate_function(argument_number, reaction_rate, reacting_meta_species, reactant_string_list, args):

        if argument_number == 0:
            return reaction_rate()
        else:
            species_arguments = []
            for meta_species, species_string, count, _ in zip(reacting_meta_species,
                                                              reactant_string_list, args, range(argument_number)):
                species_arguments.append(Species_Argument_For_Callable(meta_species, species_string, count))

        return reaction_rate(*species_arguments)

    to_return_function = \
        lambda r: rate_function(argument_number, reaction_rate, reacting_meta_species, reactant_string_list, r)

    return to_return_function, argument_number


def create_rate_function(reacting_meta_species, reactant_string_list, reaction_rate, dimension):
    if type(reaction_rate) == int or type(reaction_rate) == float or \
            isinstance(reaction_rate, Quantity):
        # Constant was used, return mass action kinetics
        reaction_rate, dimension, is_count = uh.convert_rate(reaction_rate,
                                                             len(reactant_string_list),
                                                             dimension)

        rate_function = mass_action_kinetics(reactant_string_list, reaction_rate)
        argument_number = len(reactant_string_list)

    elif isinstance(reaction_rate, ExpressionDefiner):
        reaction_rate = float(str(reaction_rate))
        rate_function = mass_action_kinetics(reactant_string_list, reaction_rate)
        argument_number = len(reactant_string_list)

    elif callable(reaction_rate):
        rate_function, argument_number = transform_callable_rate(reacting_meta_species,
                                                                 reactant_string_list, reaction_rate)

    elif type(reaction_rate) == str:
        simlog.error('Error when compiling the following reacting with the meta-species: '
                     + str(reacting_meta_species) + '\n'
                     + 'String type is not a reaction rate type not supported for this analysis')
    else:
        simlog.error('Error when compiling the following reacting with the meta-species: '
                     + str(reacting_meta_species) + '\n'
                     + str(type(reaction_rate)) + ' type is not a reaction rate type not supported for this analysis')

    return rate_function, argument_number
