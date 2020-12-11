from bs4 import BeautifulSoup as bs
import collections as col, logging as log

class Stroke(object):
    """Stroke

       This class manages input data, specifically in the form of "online" format, where the data is a time-series of {x,y} coordinates.  A stroke is conceptually part of a StrokeSet, which is a group of strokes that, together, forms a single handwriting sample.
    """

    def init(self):
        log.debug("Init")
        self.points = []

    def __init__(self):
        log.debug("Default contructor")
        self.init()

    def __init__(self, strokeXML):
        log.debug("Loader contructor")
        self.init()
        self.load(strokeXML)

    def load(self, strokeXML):
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
                x = point["x"]
                y = point["y"]
                t = point["time"]
                log.info(f"Loading point x, y, time: ({x}, {y}, {t})")
                self.points.append((x, y, t))
        except:
            log.warning(f"Could not parse point {point}", exc_info=True)