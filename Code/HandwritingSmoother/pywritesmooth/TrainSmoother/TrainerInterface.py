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
    def test():
        raise NotImplementedError

    @abstractmethod
    def getError():
        raise NotImplementedError

    @abstractmethod
    def save():
        raise NotImplementedError

    @abstractmethod
    def load():
        raise NotImplementedError