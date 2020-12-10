from bs4 import BeautifulSoup as bs
import collections as col

class Stroke(object):
    """Stroke

       This class manages input data, specifically in the form of "online" format, where the data is a time-series of {x,y} coordinates.  A stroke is conceptually part of a StrokeSet, which is a group of strokes that, together, forms a single handwriting sample.
    """

    def __init__(self):
        self.points = []

    def load(self, strokeXML):
        """load

           strokeXML is an XML of a stroke, which is a collection of points.
        """

        # Load points from stroke XML
        pointsXML = bs(str(strokeXML), 'lxml')
        points = pointsXML.find_all("point")

        for point in points:    # Enumerate points in each stroke
            x = point["x"]
            y = point["y"]
            t = point["time"]
            self.points.append((x, y, t))