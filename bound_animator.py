import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class AnimationState:
    def __init__(self, bound_array, q_array, j_matrix):
        self.bound_array = bound_array
        self.q_array = q_array
        self.j_matrix = j_matrix

    def next_step(self):
        for i in range(len(self.q_array)):
            if self.bound_array[i] > 0:
                if self.bound_array[i] - self.q_array[i] <= 0:
                    self.q_array[i] = self.q_array[i] - self.bound_array[i]
                    self.bound_array[i] = 0
                else:
                    self.bound_array[i] = self.bound_array[i] - self.q_array[i]
                    self.q_array[i] = 0

        # Perform matrix mul
        self.bound_array = np.matmul(self.bound_array, self.j_matrix)

def update(frame, state):
    state.next_step()
    line1.set_ydata(state.bound_array)
    line2.set_ydata(state.q_array)
    return line1, line2

def bound_animation(bound_array, q_array, j_matrix):
    state = AnimationState(bound_array, q_array, j_matrix)

    fig, ax = plt.subplots()

    global line1, line2
    line1, = ax.plot(state.bound_array, label='Array 1')
    line2, = ax.plot(state.q_array, label='Array 2')

    ax.legend()

    animation = FuncAnimation(fig, update, fargs=(state,), frames=100, interval=500)

    plt.show()

