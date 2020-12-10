import sys, os
from bs4 import BeautifulSoup as bs
import pywritesmooth.Data.Stroke as stroke

class StrokeSet(object):
    """StrokeSet

       This class manages input data, specifically in the form of "online" format, where the data is a time-series of {x,y} coordinates.  The stroke set is a grouping of individual strokes.  A stroke is where the user picks up the writing tool and then puts it back down for the next part of the script.
    """

    def __init__(self):
        self.strokes = []

    def load(self, inputFileName):
        """load

           inputFileName is a single file to load.  Each file will be loaded as a list of strokes.
        """

        try:
            # Load stroke information from XML file
            raw = []
            with open(inputFileName, "r") as file:
                raw = file.readlines()
                raw = "".join(raw)
                xml = bs(raw, 'lxml')

            # Extract stroke information
            allStrokeSets = xml.find_all("strokeset")
            for sset in allStrokeSets:      # Enumerate stroke sets in the file
                strokes = sset.find_all("stroke")

                for strokeXML in strokes:         # Enumerate strokes in each set
                    s = stroke.Stroke()
                    self.strokes.append(s.load(strokeXML))
        except:
            print("Exception: ", sys.exc_info())