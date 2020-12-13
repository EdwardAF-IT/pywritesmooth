import sys, os, glob, logging as log, numpy as np
from bs4 import BeautifulSoup as bs
import pywritesmooth.Data.Stroke as stroke
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import pywritesmooth.Utility.StrokeHelper as sh
import matplotlib.image as mpimg

class StrokeSet(object):
    """StrokeSet

       This class manages input data, specifically in the form of "online" format, where the data is a time-series of {x,y} coordinates.  The stroke set is a grouping of individual strokes.  A stroke is where the user picks up the writing tool and then puts it back down for the next part of the script.
    """

    def init(self):
        log.debug("Init")
        self.strokes = []
        self.onlineXMLFull = ''
        self.onslineXMLFile = ''
        self.onlineASCIIFull = ''
        self.onlineImageFull = ''

    def __init__(self):
        log.debug("Default contructor")
        self.init()

    def __init__(self, inputFileName):
        log.debug("Loader contructor")
        self.init()
        self.load(inputFileName)

    def assemblePaths(self, inputFileName):
        """assemblePaths

           Break apart the path information and reform it into paths for related data like the text and image files.  This is possible because the IAM online dataset follows a very clean and predictable naming scheme.
        """

        log.debug(f"Assembling pathnames for {inputFileName}")
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
                self.onslineXMLFile = onlineFileBase

                nameStart = inputFileName.index(onlineFileName)
                groupFolder = inputFileName[folderEnd+1:nameStart]      # i.e. a01\a01-000
                onlineImageFile = onlineFileBase + r'.tif'              # i.e. a01-000u-01.tif

                onlinePath = inputFileName[:folderStart]                # i.e. C:\Code\SMU\Capstone\Data\IAM Original
                onlineXMLFolder = folder                                # i.e. linestrokes-all\linestrokes

                onlineASCIIFile = glob.glob(os.path.join(onlinePath, onlineASCIIFolder, groupFolder, r"*.txt"))[0]  # First corresopnding text file

                self.onlineASCIIFull = onlineASCIIFile
                self.onlineImageFull = os.path.join(onlinePath, onlineImageFolder, groupFolder, onlineImageFile)

                log.debug(f"Folder: {folder}")
                log.debug(f"onlineXMLFull: {self.onlineXMLFull}")
                log.debug(f"onlineASCIIFull: {self.onlineASCIIFull}")
                log.debug(f"onlineImageFull: {self.onlineImageFull}")
                log.debug(f"onlinePath: {onlinePath}")
                log.debug(f"onlineImageFolder: {onlineImageFolder}")
                log.debug(f"groupFolder: {groupFolder}")
                log.debug(f"onlineASCIIFile: {onlineASCIIFile}")
                log.debug(f"onlineImageFile: {onlineImageFile}")

    def load(self, inputFileName):
        """load

           inputFileName is a single file to load.  Each file will be loaded as a list of strokes.
        """

        self.assemblePaths(inputFileName)

        try:
            # Load stroke information from XML file
            raw = []
            with open(inputFileName, "r") as file:
                log.info(f"Reading {inputFileName}")
                print(f"Reading {inputFileName}")
                raw = file.readlines()
                log.debug(f"Raw XML input: {raw}")
                raw = "".join(raw)
                xml = bs(raw, 'lxml')
                log.debug(f"Parsed XML input: {xml}")

            # Extract stroke information
            allStrokeSets = xml.find_all("strokeset")
            for sset in allStrokeSets:      # Enumerate stroke sets in the file
                strokes = sset.find_all("stroke")


                log.debug(f"Loading strokes {strokes}")  
                for strokeXML in strokes:         # Enumerate strokes in each set
                    self.strokes.append(stroke.Stroke(strokeXML))

            self.showStrokeset()
            self.getImage()
        except:
            log.error(f"Could not open input file {inputFileName}", exc_info=True)
            print("Exception: ", sys.exc_info())

    def getText(self):
        try:
            with open(self.onlineASCIIFull, "r") as file:
                lines = file.readlines()
                log.debug(f"Corresponding text: {lines}")
                return(lines)
        except:
            log.warning(f"Could not open corresponding ASCII text file {self.onlineASCIIFull}", exc_info=True)

    def getImage(self):
        try:
            #im = Image.open(self.onlineImageFull)
            log.info(f"Corresponding image file: {self.onlineImageFull}")
            #im.show()
            img = plt.imread(self.onlineImageFull)
            plt.imshow(img)
        except:
            log.warning(f"Could not open corresponding image file {self.onlineImageFull}", exc_info=True)

    def showStrokeset(self):
        try:
            log.debug(f"Raw strokes: {self.strokes}")

            samplePoints = []
            codes = []
            for stroke in self.strokes:  # Build parallel point and code lists from all strokes in the set
                log.debug(f"Raw stroke: {stroke}")

                points = [(x[0], x[1], x[2]) for x in stroke.getPoints()]  # Extract the points from stroke
                pointCodes = [Path.MOVETO] + list(np.repeat(Path.LINETO, len(points) - 1))  # Add the right amount of codes for current stroke

                [samplePoints.append(i) for i in points]  # Accumulate point tuples
                [codes.append(i) for i in pointCodes]     # Accumulate corresponding codes
           
            # Normalize by subtracting minimums to 0-base stroke set
            helper = sh.StrokeHelper()
            nPoints = helper.normalizePoints(samplePoints)  # Helper does the 0-basing
            ymax = nPoints.max(axis=0)[1]
            vertices = [(x[0], ymax-x[1]) for x in nPoints]      # Pull out just x,y tuples

            log.info(f"Drawing stroke set for {self.onslineXMLFile}")
            log.debug(f"Vertices: {max(vertices)[0]} {min(vertices)[1]} {vertices}")
            log.debug(f"Codes: {codes}")

            # Build path for matplotlib
            fig, ax = plt.subplots()
            path = Path(vertices, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=2)
            ax.add_patch(patch)
            ax.set_xlim(0, max(vertices)[0])
            ax.set_ylim(0, ymax)
            plt.show()
        except:
            log.error(f"Could not create stroke set display for {self.onslineXMLFile}", exc_info=True)