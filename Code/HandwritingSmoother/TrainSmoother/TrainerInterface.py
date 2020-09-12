import abc

class TrainerInterface(metaclass=abc.ABCMeta):
    """description of class"""

    def __init__(self):
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load_data_source') and 
                callable(subclass.load_data_source) and 
                hasattr(subclass, 'extract_text') and 
                callable(subclass.extract_text))

    @abc.abstractmethod
    def train():
        raise NotImplementedError

    @abc.abstractmethod
    def test():
        raise NotImplementedError

    @abc.abstractmethod
    def getError():
        raise NotImplementedError

    @abc.abstractmethod
    def save():
        raise NotImplementedError

    @abc.abstractmethod
    def load():
        raise NotImplementedError