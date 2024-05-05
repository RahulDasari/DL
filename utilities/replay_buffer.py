"""
    Imports
"""
import random
from collections import deque

"""
    Replay Buffer Class
"""
class ReplayBuffer:
    def __init__(self, buffer_size) -> None:
        self.__size = buffer_size
        self.__buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, terminated):
        """
        Adds a new experience to the replay buffer
        """
        # Add all objects to a tuple of (state, action, reward, next_state) then add it to the buffer
        self.__buffer.append((state, action, reward, next_state, terminated))
    
    def sample(self, batch_size):
        """
        Returns a list of experiences from the replay buffer randomly.
        """
        return random.sample(self.__buffer, batch_size)
    
    def empty(self):
        """
        Empties the Replay Buffer
        """
        self.__buffer.clear()

    def is_empty(self):
        """
        Checks if the Replay Buffer is empty
        """
        return len(self.__buffer) <= 0
    
    def is_full(self):
        """
        Checks if the Replay Buffer is empty
        """
        return len(self.__buffer) == self.__size
    
    def get_buffer(self):
        """
        Returns the entire Priority Replay Buffer
        """
        return self.__buffer

    def get_len(self):
        """
        Returns the length of the buffer
        """
        return len(self.__buffer)