import numpy as np, logging as log

class StrokeHelper(object):
    """StrokeHelper

       Helper methods for Stroke data types.
    """

    def init(self):
        log.debug("Init")
        self.name = "StrokeHelper"

    def __init__(self):
        log.debug("Default contructor")
        self.init()

    def normalizePoints(self, points):
        """normalizePoints

           Return the collection of stroke points as a 2D numpy array where the minimum point valules are 0.
        """

        norm = np.array(points)
        normX = norm[:,0].astype(float)
        normY = norm[:,1].astype(float)
        normT = norm[:,2].astype(float)

        new = np.empty([len(normX), 3])
        new[:,0] = normX - min(normX)
        new[:,1] = normY - min(normY)
        new[:,2] = normT - min(normT)

        log.debug(f"Normalized array: {new}")
        return(new)