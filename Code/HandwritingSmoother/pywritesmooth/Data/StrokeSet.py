# Basics
import sys, os, glob, logging as log, numpy as np

# Project
import pywritesmooth.Data.Stroke as stroke
import pywritesmooth.Utility.StrokeHelper as sh

# Parsing
from bs4 import BeautifulSoup as bs

# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.path import Path
from PIL import Image

class StrokeSet(object):
    """StrokeSet

       This class manages input data, specifically in the form of "online" format, where the 
       data is a time-series of {x,y} coordinates.  The stroke set is a grouping of individual 
       strokes.  A stroke is where the user picks up the writing tool and then puts it back down 
       for the next part of the script.

       In addition to a collection of strokesets that comprise a single handwriting line sample,
       the corresponding raster image's location from the original dataset along with the line
       of ascii text is stored.  The individual strokes are stored as a list.
    """

    def init(self):
        log.debug("Init")
        self.strokes = []       # List of online x,y point groups that comprise each sample
        self.text = ""          # The ascii version of the sample

        self.online_XML_full = ''
        self.online_XML_file = ''
        self.online_ASCII_full = ''
        self.online_image_full = ''

        self.x_offset = 1e20
        self.y_offset = 1e20
        self.y_height = 0

    def __init__(self):
        log.debug("Default constructor")
        self.init()

    def __init__(self, input_filename):
        log.debug("Loader constructor")
        self.init()
        self.load(input_filename)

    def __len__(self):
        return(len(self.strokes))

    def assemble_paths(self, input_filename):
        """assemblePaths

           Break apart the path information and reform it into paths for related data like the 
           text and image files.  This is possible because the IAM online dataset follows a very 
           clean and predictable naming scheme.
        """

        log.debug(f"Assembling pathnames for {input_filename}")
        input_filename = input_filename.lower()
        online_XML_folders = [r'linestrokes-all\linestrokes', r'original-xml-all\original', r'original-xml-part\original']
        online_ASCII_folder = r'ascii-all\ascii'
        online_image_folder = r'lineimages-all\lineimages'

        self.online_XML_full = input_filename   # Full XML file with path

        for folder in online_XML_folders:
            if folder in input_filename:
                folder_start = input_filename.index(folder)
                folder_end = folder_start + len(folder)

                online_filename = os.path.split(input_filename)[1]        # i.e. a01-000u-01.xml
                line_number = int(online_filename[-6:-4]) - 1            # i.e. 01                
                online_file_base = online_filename[:-4]                    # i.e. a01-000u-01
                self.online_XML_file = online_file_base

                name_start = input_filename.index(online_filename)
                group_folder = input_filename[folder_end+1:name_start]      # i.e. a01\a01-000
                online_image_file = online_file_base + r'.tif'              # i.e. a01-000u-01.tif

                online_path = input_filename[:folder_start]                # i.e. C:\Code\SMU\Capstone\Data\IAM Original
                online_XML_folder = folder                                # i.e. linestrokes-all\linestrokes

                online_ASCII_file = glob.glob(os.path.join(online_path, online_ASCII_folder, group_folder, r"*.txt"))[0]  # First corresopnding text file

                self.online_ASCII_full = online_ASCII_file
                self.online_image_full = os.path.join(online_path, online_image_folder, group_folder, online_image_file)
                self.text = self.load_text(line_number)

                log.debug(f"Folder: {folder}")
                log.debug(f"online_XML_full: {self.online_XML_full}")
                log.debug(f"online_ASCII_full: {self.online_ASCII_full}")
                log.debug(f"online_image_full: {self.online_image_full}")
                log.debug(f"online_path: {online_path}")
                log.debug(f"online_image_folder: {online_image_folder}")
                log.debug(f"group_folder: {group_folder}")
                log.debug(f"online_ASCII_file: {online_ASCII_file}")
                log.debug(f"online_image_file: {online_image_file}")

    def load(self, input_filename):
        """load

           input_filename is a single file to load.  Each file will be loaded as a list of strokes.
           In addition, offsets are calculated and saved for processing if they are needed.
        """

        self.assemble_paths(input_filename)

        try:
            # Load stroke information from XML file
            raw = []
            with open(input_filename, "r") as file:
                log.info(f"Reading {input_filename}")
                print(f"Reading {input_filename}")
                raw = file.readlines()
                log.debug(f"Raw XML input: {raw}")
                raw = "".join(raw)
                xml = bs(raw, 'lxml')
                log.debug(f"Parsed XML input: {xml}")
        except:
            log.error(f"Could not open input file {input_filename}", exc_info=True)
            print("Exception: ", sys.exc_info())
            return

        try:
            # Get offset information
            stroke_offsets = xml.find("whiteboarddescription")
            diag_offset = stroke_offsets.find("diagonallyoppositecoords")
            vert_offset = stroke_offsets.find("verticallyoppositecoords")
            hori_offset = stroke_offsets.find("horizontallyoppositecoords")

            # Calculate offsets
            self.x_offset = min(self.x_offset, int(diag_offset["x"]), int(vert_offset["x"]), int(hori_offset["x"]))
            self.y_offset = min(self.y_offset, int(diag_offset["y"]), int(vert_offset["y"]), int(hori_offset["y"]))
            self.y_height = max(self.y_height, int(diag_offset["y"]), int(vert_offset["y"]), int(hori_offset["y"]))
            self.y_height -= self.y_offset
            self.x_offset -= 100
            self.y_offset -= 100

            # Extract stroke information
            all_strokesets = xml.find_all("strokeset")
            for sset in all_strokesets:      # Enumerate stroke sets in the file
                strokes = sset.find_all("stroke")

                log.debug(f"Loading strokes {strokes}")  
                for stroke_XML in strokes:         # Enumerate strokes in each set
                    self.strokes.append(stroke.Stroke(stroke_XML, self.x_offset, self.y_offset))

            #self.get_image()
            #self.show_strokeset()
        except:
            log.error(f"Could not process file {inputFileName}", exc_info=True)
            print("Exception: ", sys.exc_info())

    def as_delta_array(self):
        """asDeltaArray

           Convert strokes, which are stored in absolute coordinates, into a numpy array of the 
           differences between each point.  Each row is [delta_x, delta_y, strokeEndFlag].
           Adapted from https://github.com/adeboissiere/Handwriting-Prediction-and-Synthesis.
        """

        # Initialize a zero matrix of the right size
        n_point = 0
        for i in range(len(self.strokes)):
            n_point += len(self.strokes[i])
        stroke_data = np.zeros((n_point, 3), dtype=np.int16)
        log.debug(f"Convert to array input: Len={len(stroke_data)} Strokes={stroke_data}")

        # Compute the deltas and save in numpy matrix
        prev_x = 0
        prev_y = 0
        counter = 0
        for j in range(len(self.strokes)):          # Enumerate strokes
            points = self.strokes[j].get_points()
            for k in range(len(points)):            # Enumerate points in each stroke
                log.debug(f"Convert to array stroke data: j={j}, k={k}, {counter}, x={int(points[k][0])}, y={int(points[k][1])}, prev_x={prev_x}, prev_y={prev_y}")
                stroke_data[counter, 0] = int(points[k][0]) - prev_x
                stroke_data[counter, 1] = int(points[k][1]) - prev_y
                log.debug(f"Convert to array delta: {stroke_data[counter, 0]} {stroke_data[counter, 1]}")
                    
                prev_x = int(points[k][0])
                prev_y = int(points[k][1])
                stroke_data[counter, 2] = 0
                if (k == (len(self.strokes[j])-1)): # End of stroke
                    stroke_data[counter, 2] = 1
                counter += 1

        log.debug(f"Convert to array output: {self.array_to_string(stroke_data)}")
        return stroke_data

    def array_to_string(self, arr):
        """array_to_string

           Convert a numpy array into a string.  Useful to print an entire array instead of the
           truncation that numpy will give you.
        """

        s = ""

        for i in range(len(arr)):
            s += np.array_str(arr[i]) + '\n'

        return(s)

    def load_text(self, line_number):
        """loadText

           Read the ascii text that corresponds to the handwriting stroke information.  The
           input dataset is well-structured, so this is easy to do with the right filename
           and the line number of interest.  The ascii files have 2 sections.  The first is an OCR
           attempt at interpreting the handwriting, and the second, starting with CSR, is the
           correct text.  We read in only the correct text.
        """

        try:
            with open(self.online_ASCII_full, "r") as file:
                s = file.read()

            s = s[s.find("CSR"):]       # The second and more accurate text block

            if len(s.split("\n")) > line_number+2:
                s = s.split("\n")[line_number+2]
                log.debug(f"Corresponding text: {s}")
                return s
        except:
            log.warning(f"Could not open corresponding ASCII text file {self.online_ASCII_full}", exc_info=True)

    def get_text(self):
        """get_text

           Return the ascii text that represents these strokes.
        """

        return self.text

    def get_image(self):
        """get_image

           Show the raster image that corresponds to the writing sample.  This image is stored in the
           input dataset following a similar naming scheme to the ascii text.
        """

        try:
            log.info(f"Corresponding image file: {self.online_image_full}")
            #im.show()
            img = plt.imread(self.online_image_full)
            #plt.imshow(img)
        except:
            log.warning(f"Could not open corresponding image file {self.online_image_full}", exc_info=True)

    def show_strokeset(self):
        """show_strokeset

           Display the handwriting sample by plotting the online data points.  This is done by 
           extracting the x,y points from a numpy version of the data, then applying a LINETO
           action from matplotlib to connect the dots.  A MOVETO action is used when the virtual
           pen is lifted between strokes, which results in the white space expected.

           In addition, the points are normalized in order to fit neatly on the drawing surface,
           and the aspect ratio is honored so that the sample appears exactly as it was written.
           Finally, all the usual chrome of a plot, like labels, axes, and ticks are removed
           since they make no sense in this context.
        """

        try:
            log.debug(f"Raw strokes: {self.strokes}")

            sample_points = []
            codes = []
            for stroke in self.strokes:  # Build parallel point and code lists from all strokes in the set
                log.debug(f"Raw stroke: {stroke}")

                points = [(x[0], x[1], x[2]) for x in stroke.as_numpy_array()]  # Extract the points from stroke
                point_codes = [Path.MOVETO] + list(np.repeat(Path.LINETO, len(points) - 1))  # Add the right amount of codes for current stroke

                [sample_points.append(i) for i in points]  # Accumulate point tuples
                [codes.append(i) for i in point_codes]     # Accumulate corresponding codes
           
            # Normalize by subtracting minimums to 0-base stroke set
            helper = sh.StrokeHelper()
            nPoints = helper.normalize_points(sample_points)  # Helper does the 0-basing
            ymax = nPoints.max(axis=0)[1]                   # Sum the 3 rows in nPoints and take the second, which is y
            vertices = [(x[0], ymax-x[1]) for x in nPoints]      # Pull out just x,y tuples where y is subtracted from its max to appear right-side up
            xmax = max(vertices)[0]                           # X is the 0th element of the tuples in vertices, find its maximum

            log.info(f"Drawing stroke set for {self.online_XML_file}")
            log.debug(f"Vertices: {max(vertices)[0]} {min(vertices)[1]} {vertices}")
            log.debug(f"Codes: {codes}")

            # Build path for matplotlib
            path = Path(vertices, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=2)

            # Plot for rendering
            fig, ax = plt.subplots(figsize = (8, 2), facecolor='w', edgecolor='k')
            fig.canvas.set_window_title(f"Online rendering for {self.online_XML_file}")

            ax.add_patch(patch)     # Point plot of the handwriting
            ax.set_aspect('equal')  # Lock the aspect ratio so writing looks like it did originally
            ax.set_xlim(0, xmax)    # Size plot to the dimensions of the handwriting
            ax.set_ylim(0, ymax)
            ax.axis('off')          # Turn off plot axes since it isn't really a traditional plot
            plt.tight_layout()      # Trim excess window whitespace
            plt.show()
        except:
            log.error(f"Could not create stroke set display for {self.online_XML_file}", exc_info=True)