from bs4 import BeautifulSoup as bs
import collections as col, logging as log
import numpy as np
import pywritesmooth.Utility.StrokeHelper as sh

class Stroke(object):
    """Stroke

       This class manages input data, specifically in the form of "online" format, where the data is a time-series of {x,y} coordinates.  A stroke is conceptually part of a StrokeSet, which is a group of strokes that, together, forms a single handwriting sample.
    """

    def init(self):
        log.debug("Init")
        self.orig_points = []       # Raw from the input file
        self.points = []            # After any offsets are applied

    def __init__(self):
        log.debug("Default constructor")
        self.init()

    def __init__(self, strokeXML, x_offset = 0, y_offset = 0):
        log.debug("Loader constructor")
        self.init()
        self.load(strokeXML, x_offset, y_offset)

    def __len__(self):
        return(len(self.getPoints()))

    def load(self, strokeXML, x_offset = 0, y_offset = 0):
        """load

           strokeXML is an XML of a stroke, which is a collection of points.
        """

        # Load points from stroke XML
        log.debug(f"Loading from raw stroke: {strokeXML}")
        pointsXML = bs(str(strokeXML), 'lxml')
        log.debug(f"Stroke parsed XML: {pointsXML}")
        points = pointsXML.find_all("point")
        log.debug(f"Loading points: {points}")

        try:
            for point in points:    # Enumerate points in each stroke
                x = int(point["x"])
                y = int(point["y"])
                t = float(point["time"])
                log.debug(f"Loading point x, y, time: ({x}, {y}, {t})")
                log.debug(f"Offset point x, y, time: ({x - x_offset}, {y - y_offset}, {t})")
                self.orig_points.append((x, y, t))
                self.points.append((x - x_offset, y - y_offset, t))
        except:
            log.warning(f"Could not parse point {point}", exc_info=True)

    def asNumpyArray(self):
        """asNumpyArray

           Return the collection of stroke points as a 2D numpy array.
        """

        return(np.array(self.points))

    def getPoints(self):
        """getPoints

           Return the collection of stroke points.
        """

        return(self.points)

    def getNormalizedPoints(self):
        """getNormalizedPoints

           Return the collection of stroke points as a 2D numpy array where the minimum point valules are 0.
        """

        helper = sh.StrokeHelper()
        return(helper.normalizePoints(self.points))