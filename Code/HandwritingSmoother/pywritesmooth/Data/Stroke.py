# Basics
import collections as col, numpy as np, logging as log

# Project
import pywritesmooth.Utility.StrokeHelper as sh

# Parsing
from bs4 import BeautifulSoup as bs

class Stroke(object):
    """Stroke

       This class manages input data, specifically in the form of "online" format, where the 
       data is a time-series of {x,y} coordinates.  A stroke is conceptually part of a 
       StrokeSet, which is a group of strokes that, together, forms a single handwriting 
       sample.

       In order to facilitate later model training, offsets are also calculated and stored.
       The offsets are computed when the dataset is being loaded in.  They come from the input
       file a level above this one (the stroke), at the strokeset level.  Therefore, the strokeset
       to which this stroke belongs is responsible for reading the offsets, computing them, and
       passing them into this object as needed.

       The normalized points are saved along with the original, non-normalized, points to maintain
       flexibility for users of this class.
    """

    def init(self):
        log.debug("Init")
        self.orig_points = []       # Raw from the input file
        self.points = []            # After any offsets are applied

    def __init__(self):
        log.debug("Default constructor")
        self.init()

    def __init__(self, stroke_XML, x_offset = 0, y_offset = 0):
        log.debug("Loader constructor")
        self.init()
        self.load(stroke_XML, x_offset, y_offset)

    def __len__(self):
        return(len(self.get_points()))

    def load(self, stroke_XML, x_offset = 0, y_offset = 0):
        """load

           strokeXML is an XML of a stroke, which is a collection of points.  In addition, if
           offsets are passed in, they will be calculated and saved with the stroke, too.
        """

        # Load points from stroke XML
        log.debug(f"Loading from raw stroke: {stroke_XML}")
        points_XML = bs(str(stroke_XML), 'lxml')
        log.debug(f"Stroke parsed XML: {points_XML}")
        points = points_XML.find_all("point")
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

    def as_numpy_array(self):
        """asNumpyArray

           Return the collection of stroke points as a 2D numpy array.
        """

        return(np.array(self.points))

    def get_points(self):
        """getPoints

           Return the collection of stroke points.
        """

        return(self.points)

    def getNormalizedPoints(self):
        """getNormalizedPoints

           Return the collection of stroke points as a 2D numpy array where the minimum point valules are 0.
        """

        helper = sh.StrokeHelper()
        return(helper.normalize_points(self.points))