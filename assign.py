
class Assign:

    def __init__(self, assignment, *spe):

        self.assignment = assignment
        self.species = spe

    def evaluate(self, argument_dict):

        count_list = [0 for _ in range(len(self.species))]
        for key in argument_dict:
            i = self.species.index(key)
            count_list[i] = argument_dict[key]

        return self.assignment(*count_list)




