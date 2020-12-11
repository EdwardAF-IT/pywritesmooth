import sys, os
import pywritesmooth.Data.StrokeSet as strokeset

class StrokeDataset(object):
    """StrokeDataset

       This is a collection of stroke sets, where each stroke set is an individual sample of handwriting comprised of a series of strokes.  A stroke is the set of points from the time the writing implement in placed on the surface until it is lifted up.
    """

    def init(self):
        self.strokesets = []

    def __init__(self):
        self.init()

    def __init__(self, inputFiles):
        self.init()
        self.load(inputFiles)
        
    def load(self, inputFiles):
        """load

           inputFiles is a list of files to load.  Each file will be loaded as a strokeset.
        """
        
        # Load stroke information from XML files
        for file in inputFiles:
            self.strokesets.append(strokeset.StrokeSet(file))