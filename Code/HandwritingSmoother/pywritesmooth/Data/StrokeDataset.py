import sys, os, pickle, logging as log
import numpy as np
import pywritesmooth.Data.StrokeSet as strokeset
import pywritesmooth.Utility.StrokeHelper as sh

class StrokeDataset(object):
    """StrokeDataset

       This is a collection of stroke sets, where each stroke set is an individual sample of handwriting comprised of a series of strokes.  A stroke is the set of points from the time the writing implement in placed on the surface until it is lifted up.
    """

    def init(self):
        log.debug("Init")
        self.strokesets = []        # L:ist of strokeset objects
        self.strokeMatrix = []      # List of strokeset matrices
        self.strokeAscii = []       # List of text lines

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

        log.debug(f"Stroke Sets: Len = {len(self.getStrokesets())}")
        log.debug(f"Stroke Matrix: Len = {len(self.getStrokeMatrix())}, Strokes = {self.getStrokeMatrix()}")
        log.debug(f"Ascii Matrix: Len = {len(self.getAsciiList())}, Lines = {self.getAsciiList()}")

    def __len__(self):
        return(len(self.getStrokesets()))
        
    def loadRawData(self, inputFiles):
        """load

           inputFiles is a list of files to load.  Each file will be loaded as a strokeset.
        """

        log.debug(f"Loading dataset {inputFiles}")  
        print(f"Loading dataset")

        # Load stroke information from XML files
        for file in inputFiles:
            newStrokeset  = strokeset.StrokeSet(file)
            self.strokesets.append(newStrokeset)
            self.strokeMatrix.append(newStrokeset.asDeltaArray())
            self.strokeAscii.append(newStrokeset.getText())

        doneMsg = "Finished parsing dataset. Imported {} lines".format(len(self.getStrokesets()))
        print (doneMsg)
        log.info(doneMsg)

    def getStrokesets(self):
        return self.strokesets

    def getStrokeMatrix(self):
        return self.strokeMatrix

    def getAsciiList(self):
        return self.strokeAscii

    def save(self, pFile):
        saveMsg = f"Saving data as Python pickle: {pFile}"
        log.info(saveMsg)
        print(saveMsg)
        file = open(pFile, 'wb')
        pickle.dump(self, file)
        file.close()

    def load(self, pFile):
        loadMsg = f"Loading previously saved dataset from pickle file {pFile}"
        log.info(loadMsg)
        print(loadMsg)
        file = open(pFile, 'rb')
        incoming = pickle.load(file)
        file.close()

        self.strokesets = incoming.strokesets
        self.strokeMatrix = incoming.strokeMatrix
        self.strokeAscii = incoming.strokeAscii

        doneMsg = "Loaded {} lines for processing".format(len(self.getStrokesets()))
        print (doneMsg)
        log.info(doneMsg)