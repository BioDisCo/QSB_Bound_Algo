from mobspy.modules.mobspy_expressions import Specific_Species_Operator
from mobspy import BaseSpecies, New

class Species_Argument_For_Callable(Specific_Species_Operator):

    def __init__(self, meta_species, species_string, count):
        super(Species_Argument_For_Callable, self).__init__(species_string, meta_species)
        self.count = count

    def __add__(self, other):
        if isinstance(other, Species_Argument_For_Callable):
            return self.count + other.count
        else:
            return self.count + other

    def __radd__(self, other):
        if isinstance(other, Species_Argument_For_Callable):
            return self.count + other.count
        else:
            return self.count + other

    def __sub__(self, other):
        if isinstance(other, Species_Argument_For_Callable):
            return self.count - other.count
        else:
            return self.count - other

    def __rsub__(self, other):
        if isinstance(other, Species_Argument_For_Callable):
            return self.count - other.count
        else:
            return self.count - other

    def __mul__(self, other):
        if isinstance(other, Species_Argument_For_Callable):
            return self.count*other.count
        else:
            return self.count*other

    def __rmul__(self, other):
        if isinstance(other, Species_Argument_For_Callable):
            return self.count*other.count
        else:
            return self.count*other

    def __truediv__(self, other):
        if isinstance(other, Species_Argument_For_Callable):
            return self.count/other.count
        else:
            return self.count/other

    def __rtruediv__(self, other):
        if isinstance(other, Species_Argument_For_Callable):
            return other.count/self.count
        else:
            return other/self.count

    def __pow__(self, other):
        if type(other) != int and type(other) != float:
            raise TypeError('Power must only be int or floar')

        if isinstance(other, Species_Argument_For_Callable):
            return self.count**other.count
        else:
            return self.count**other

    def set_count(self, count):
        self.count = count


if __name__ == '__main__':

    A = BaseSpecies()
    A.a1, A.a2
    C = New(A)
    C.c1, C.c2

    r = Species_Argument_For_Callable(C, 'C_dot_c1_dot_a1', 10)
    r2 = Species_Argument_For_Callable(C, 'C_dot_c1_dot_a2', 10)

    if r.a1:
        print('Yes')
    if r.c2:
        print('No')

    print(10*r)


    def rate_function_example(a1, a2):

        if a1.c1:
            return 10*a1*a2
        else:
            return 1*a1

    print(rate_function_example(r, r2))

