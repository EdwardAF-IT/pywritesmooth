# Basics
import sys, os, pickle, numpy as np, logging as log

# Project
import pywritesmooth.Data.StrokeSet as strokeset
import pywritesmooth.Utility.StrokeHelper as sh

class StrokeDataset(object):
    """StrokeDataset

       This is a collection of stroke sets, where each stroke set is an individual sample 
       of handwriting comprised of a series of strokes.  A stroke is the set of points 
       from the time the writing implement in placed on the surface until it is lifted up.

       This object contains not only its corresponding strokeset objects; it also computes and
       saves those same strokesets as a list of numpy matrices, which is then easily consumable
       when training a model.  In addition, a list of the ascii lines corresponding to the
       strokesets is also saved for convenience.

       The entire dataset is saved to disk when it has been read in using the Python pickle
       library.  This saves a tremendous amount of time on subsequent runs by first checking
       to see if a saved file is available and loading it if so.  Because the file is already
       in native Python format and saved exactly as the data structures represented by these
       objects, it is significantly faster than processing the raw input data.
    """

    def init(self):
        log.debug("Init")
        self.strokesets = []        # L:ist of strokeset objects
        self.stroke_matrix = []      # List of strokeset matrices
        self.stroke_ascii = []       # List of text lines

    def __init__(self):
        log.debug("Default constructor")
        self.init()

    def __init__(self, input_files, saved_pickle = None):
        self.init()

        if not saved_pickle is None:    # Only load/save if a filename is provided
            if os.path.exists(saved_pickle):
                self.load(saved_pickle)
            else:
                self.load_raw_data(input_files)
                self.save(saved_pickle)
        else:
            self.load_raw_data(input_files)

        log.debug(f"Stroke Sets: Len = {len(self.get_strokesets())}")
        log.debug(f"Stroke Matrix: Len = {len(self.get_stroke_matrix())}, Strokes = {self.get_stroke_matrix()}")
        log.debug(f"Ascii Matrix: Len = {len(self.get_ascii_list())}, Lines = {self.get_ascii_list()}")

    def __len__(self):
        return(len(self.get_strokesets()))
        
    def load_raw_data(self, input_files):
        """load

           input_files is a list of files to load.  Each file will be loaded as a strokeset.
           In addition, each stroke set is asked to compute itself as a numpy array.  That array
           and its corresponding ascii text is saved as lists for easy consumption by trainers.
        """

        log.debug(f"Loading dataset {input_files}")  
        print(f"Loading dataset")

        # Load stroke information from XML files
        for file in input_files:
            new_strokeset  = strokeset.StrokeSet(file)
            self.strokesets.append(new_strokeset)
            self.stroke_matrix.append(new_strokeset.as_delta_array())
            self.stroke_ascii.append(new_strokeset.get_text())

        done_msg = "Finished parsing dataset. Imported {} lines".format(len(self.get_strokesets()))
        print (done_msg)
        log.info(done_msg)

    def get_strokesets(self):
        return self.strokesets

    def get_stroke_matrix(self):
        return self.stroke_matrix

    def get_ascii_list(self):
        return self.stroke_ascii

    def save(self, pFile):
        """save

           Write the entire dataset to a Python pickle file for later retrieval.
        """

        save_msg = f"Saving data as Python pickle: {pFile}"
        log.info(save_msg)
        print(save_msg)
        os.makedirs(os.path.dirname(pFile), exist_ok=True)
        file = open(pFile, 'wb')
        pickle.dump(self, file)
        file.close()

    def load(self, pFile):
        """load

           Load the entire dataset from a Python pickle file into this object.  Loading a
           previously saved file is much faster than reading the raw input data.
        """

        load_msg = f"Loading previously saved dataset from pickle file {pFile}"
        log.info(load_msg)
        print(load_msg)
        file = open(pFile, 'rb')
        incoming = pickle.load(file)
        file.close()

        # The collections must be assigned individually like this since the loaded object loses scope
        # upon exiting this method.
        self.strokesets = incoming.strokesets
        self.stroke_matrix = incoming.stroke_matrix
        self.stroke_ascii = incoming.stroke_ascii

        done_msg = "Loaded {} lines for processing".format(len(self.get_strokesets()))
        print (done_msg)
        log.info(done_msg)