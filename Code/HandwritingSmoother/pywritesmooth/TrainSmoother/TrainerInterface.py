import logging as log
from abc import ABCMeta, abstractmethod

class TrainerInterface(metaclass=ABCMeta):
    """description of class"""

    def __init__(self):
        print("In TI con")

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load_data_source') and 
                callable(subclass.load_data_source) and 
                hasattr(subclass, 'extract_text') and 
                callable(subclass.extract_text))

    @abstractmethod
    def train():
        raise NotImplementedError

    @abstractmethod
    def train_network():
        raise NotImplementedError

    @abstractmethod
    def loss_fn():
        raise NotImplementedError