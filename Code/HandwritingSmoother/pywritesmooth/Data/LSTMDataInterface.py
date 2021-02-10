# Basics
import numpy as np, logging as log
import reprlib

class LSTMDataInterface():
    """  LSTMDataInterface

        Wrapper for a single stroke dataset.  Serves as an interface between the data and the model
        by providing data in the model's expected format.  This class relies on being given a loaded
        instance of StrokeDataset in order to obtain the data that it needs to manage for the trainer.

        Adapted from: https://github.com/adeboissiere/Handwriting-Prediction-and-Synthesis
    """

    def __init__(self, train_strokeset, batch_size=50, tsteps=300, scale_factor = 10, U_items=10, limit = 500, alphabet="default"):
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.tsteps = tsteps
        self.scale_factor = scale_factor # Divide data by this factor
        self.limit = limit # Removes large noisy gaps in the data
        self.U_items = U_items

        self.load_preprocessed(train_strokeset)
        self.reset_batch_pointer()

    def load_preprocessed(self, train_strokeset):
        """load_preprocessed

           The key to the whole class is in here; it must be able to obtain the stroke data in the
           form of a numpy matrix that can be consumed by the trainer.   In addition, it must also
           be able to get a list of the corresponding ascii text for those stroke samples.

           With the needed data in hand, it is processed for scaling and removing large gaps.  The
           final form is then appended to lists of stroke and ascii data, ready for use by the trainer.

        """
        # Get data in the loader's required format
        self.raw_stroke_data = train_strokeset.get_stroke_matrix()
        self.raw_ascii_data = train_strokeset.get_ascii_list()

        # Goes thru the list and only keeps the text entries that have more than tsteps points
        self.stroke_data = []
        self.ascii_data = []
        counter = 0

        for i in range(len(self.raw_stroke_data)):
            data = self.raw_stroke_data[i]
            if len(data) > (self.tsteps+2):
                # Removes large gaps from the data
                data = np.minimum(data, self.limit)
                data = np.maximum(data, -self.limit)
                data = np.array(data,dtype=np.float32)
                data[:,0:2] /= self.scale_factor
                
                self.stroke_data.append(data)
                self.ascii_data.append(self.raw_ascii_data[i])

        # Minus 1, since we want the ydata to be a shifted version of x data
        self.num_batches = int(len(self.stroke_data) / self.batch_size)
        log.info(f"Stroke Len = {len(self.stroke_data)}, Batch Size = {self.batch_size}, Num Batches = {self.num_batches}")
        print ("Loaded dataset:")
        print ("   -> {} individual data points".format(len(self.stroke_data)))
        print ("   -> {} batches".format(self.num_batches))

    def next_batch(self):
        """next_batch
           
           Returns a randomized, tsteps sized portion of the training data.  This is a batch that
           is to be processed as a batch by the trainer.
        """

        x_batch = []
        y_batch = []
        ascii_list = []
        for i in range(self.batch_size):
            data = self.stroke_data[self.idx_perm[self.pointer]]
            x_batch.append(np.copy(data[:self.tsteps]))
            y_batch.append(np.copy(data[1:self.tsteps+1]))
            ascii_list.append(self.ascii_data[self.idx_perm[self.pointer]])
            self.tick_batch_pointer()
        one_hots = [self.one_hot(s) for s in ascii_list]
        return x_batch, y_batch, ascii_list, one_hots
    
    def one_hot(self, s):
        """one_hot

           Transforms a string sequence into a one-hot matrix. Dimensions of the output one-hot 
           matrix are (string length, len(alphabet)).
        """

        # Index position 0 means "unknown"
        if self.alphabet == "default":
            alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

        if s is None:
            seq = [0]
            log.debug(f"One hotting: nothing")
        else:
            seq = [alphabet.find(char) + 1 for char in s]
            log.debug(f"One hotting: {s}")

        if len(seq) >= self.U_items:
            seq = seq[:self.U_items]
        else:
            seq = seq + [0]*(self.U_items - len(seq))
        one_hot = np.zeros((self.U_items,len(alphabet)+1))
        one_hot[np.arange(self.U_items),seq] = 1

        return one_hot

    def tick_batch_pointer(self):
        """tick_batch_pointer

           Increment to the next batch.  If we've exhausted all available batches, then reset
           the pointer.
        """

        self.pointer += 1
        if (self.pointer >= len(self.stroke_data)):
            self.reset_batch_pointer()

    def reset_batch_pointer(self):
        """reset_batch_pointer

           The batch pointer keeps track of which batch is being processed for the trainer's use.
           This method will reset that pointer and select a new random stroke for the trainer
           to use.
        """

        self.idx_perm = np.random.permutation(len(self.stroke_data))
        self.pointer = 0
        log.debug("Pointer reset")