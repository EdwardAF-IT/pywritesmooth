import sys, os, glob
from bs4 import BeautifulSoup as bs
import pywritesmooth.Data.Stroke as stroke
from PIL import Image

class StrokeSet(object):
    """StrokeSet

       This class manages input data, specifically in the form of "online" format, where the data is a time-series of {x,y} coordinates.  The stroke set is a grouping of individual strokes.  A stroke is where the user picks up the writing tool and then puts it back down for the next part of the script.
    """

    def init(self):
        self.strokes = []
        self.onlineXMLFull = ''
        self.onlineASCIIFull = ''
        self.onlineImageFull = ''

    def __init__(self):
        self.init()

    def __init__(self, inputFileName):
        self.init()
        self.load(inputFileName)

    def assemblePaths(self, inputFileName):
        """assemblePaths

           Break apart the path information and reform it into paths for related data like the text and image files.  This is possible because the IAM online dataset follows a very clean and predictable naming scheme.
        """
        inputFileName = inputFileName.lower()
        onlineXMLFolders = [r'linestrokes-all\linestrokes', r'original-xml-all\original', r'original-xml-part\original']
        onlineASCIIFolder = r'ascii-all\ascii'
        onlineImageFolder = r'lineimages-all\lineimages'

        self.onlineXMLFull = inputFileName   # Full XML file with path

        for folder in onlineXMLFolders:
            if folder in inputFileName:
                folderStart = inputFileName.index(folder)
                folderEnd = folderStart + len(folder)

                onlineFileName = os.path.split(inputFileName)[1]        # i.e. a01-000u-01.xml
                onlineFileBase = onlineFileName[:-4]                    # i.e. a01-000u-01

                nameStart = inputFileName.index(onlineFileName)
                groupFolder = inputFileName[folderEnd+1:nameStart]      # i.e. a01\a01-000
                onlineImageFile = onlineFileBase + r'.tif'              # i.e. a01-000u-01.tif

                onlinePath = inputFileName[:folderStart]                # i.e. C:\Code\SMU\Capstone\Data\IAM Original
                onlineXMLFolder = folder                                # i.e. linestrokes-all\linestrokes

                onlineASCIIFile = glob.glob(os.path.join(onlinePath, onlineASCIIFolder, groupFolder, r"*.txt"))[0]  # First corresopnding text file

                self.onlineASCIIFull = onlineASCIIFile
                self.onlineImageFull = os.path.join(onlinePath, onlineImageFolder, groupFolder, onlineImageFile)

    def load(self, inputFileName):
        """load

           inputFileName is a single file to load.  Each file will be loaded as a list of strokes.
        """

        self.assemblePaths(inputFileName)

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
                    self.strokes.append(stroke.Stroke(strokeXML))
        except:
            print("Exception: ", sys.exc_info())

    def getText(self):
        with open(self.onlineASCIIFull, "r") as file:
            return(file.readlines())

    def getImage(self):
        im = Image.open(self.onlineImageFull)
        im.show()