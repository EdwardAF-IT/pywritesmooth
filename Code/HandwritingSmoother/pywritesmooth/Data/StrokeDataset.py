import sys, os, pickle, logging as log
import pywritesmooth.Data.StrokeSet as strokeset
import pywritesmooth.Utility.StrokeHelper as sh

class StrokeDataset(object):
    """StrokeDataset

       This is a collection of stroke sets, where each stroke set is an individual sample of handwriting comprised of a series of strokes.  A stroke is the set of points from the time the writing implement in placed on the surface until it is lifted up.
    """

    def init(self):
        log.debug("Init")
        self.strokesets = []

    def __init__(self):
        log.debug("Default constructor")
        self.init()

    def __init__(self, inputFiles, savedPickle):
        log.debug("Loader constructor")
        self.init()

        if os.path.exists(savedPickle):
            self.load(savedPickle)
        else:
            self.loadRawData(inputFiles)
            self.save(savedPickle)
        
    def loadRawData(self, inputFiles):
        """load

           inputFiles is a list of files to load.  Each file will be loaded as a strokeset.
        """

        log.debug(f"Loading dataset {inputFiles}")  
        print(f"Loading dataset")

        # Load stroke information from XML files
        for file in inputFiles:
            self.strokesets.append(strokeset.StrokeSet(file))

    def save(self, pFile):
        log.info(f"Saving data as Python pickle: {pFile}")
        file = open(pFile, 'wb')
        pickle.dump(self, file)
        file.close()

    def load(self, pFile):
        log.info(f"Loading data from Python pickle: {pFile}")
        print(f"Loading previously saved dataset from pickle {pFile}")
        file = open(pFile, 'rb')
        self = pickle.load(file)
        file.close()