# Basics
import os, time, svgwrite, platform, numpy as np, logging as log

# Project
from .TrainerInterface import TrainerInterface
from .HandwritingSynthesisModel import HandwritingSynthesisModel
from pywritesmooth.Data.LSTMDataInterface import LSTMDataInterface

# Neural Networks
import torch
from torch import nn, optim
use_cuda = False
#use_cuda = torch.cuda.is_available()

# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from IPython.display import display, SVG
is_linux = True if platform.system().lower() == 'linux' else False

class LSTMTrainer(TrainerInterface):
    """LSTMTrainer

       This class is the main driver to train an LSTM handwriting model.  It depends
       on an implement model from torch.nn.Module; in this case, it uses the
       custom HandwritingSynthesisModel class.  In addition, it also needs an interface
       to the data that will feed the trainer batches and manage the data pointers.  This
       is provided with the LSTMDataInterface class, which bridges the gap between the
       main data structure (given by StrokeDataset) and the needs of this trainer.

       Common and required parameters are given meaningful defaults in the constructor.
       They can be overridden, of course, but this gives us a good place to start.

       The main driver of the class is the train() method.  This is a required method of
       the superclass.  In this case, train() is used to instantiate our 
       HandwritingSynthesisModel.  Then, if a trained model is available in a saved file,
       that model is loaded.  This saves a tremendous amount of time once the model has
       been trained.  If no model is available, then it is trained and saved.  (Hint: use
       GPUs!)

       Training the model is managed by the train_network() method.  It loops over the 
       specified number of epochs and batches using a custom Pytorch LSTM network (from
       HandwritingSynthesisModel).  The loss is computed by following the Gaussian mixture
       equations from the paper.

       Finally, there are methods to optionally create plots of the sample strokes and
       heatmaps from the original paper upon which this model implementation is based.
       You can read the original paper here:  https://arxiv.org/pdf/1308.0850v5.pdf

       Adapted from: https://github.com/adeboissiere/Handwriting-Prediction-and-Synthesis
    """
    def init(self):
        # Preferences
        self.display_images = False
        self.save_plot_base = os.path.join(".", "plots", "phi")
        self.plot_num = 1
        self.hw_plot_base = os.path.join(".", "samples", "hw")
        self.hw_num = 1
        self.gen_stroke_base = os.path.join(".", "strokes", "gen_stroke")
        self.gen_stroke_num = 1

        # Batch params
        self.n_batch = 20
        self.sequence_length = 400
        self.U_items = int(self.sequence_length/25)
        self.eps = float(np.finfo(np.float32).eps)

        # Network params
        self.hidden_size = 256
        self.n_layers = 3
        self.n_gaussians = 20
        self.Kmixtures = 10
        self.trained_model = None

        # Hyperparameters
        self.gradient_threshold = 10
        self.dropout = 0.2
        self.epoch = 10

    def __init__(self, saved_model = None, display_images = False, save_plot_base = None,
                 save_samples = False, save_sample_base = None, 
                 save_generated_strokes = False, save_generated_stroke_base = None,
                 epoch = 10):
        log.debug("In ltsm con")
        self.init()
        
        self.epoch = epoch

        self.display_images = display_images
        if not save_plot_base is None:
            self.save_plot_base = save_plot_base

        self.save_samples = save_samples
        if not save_sample_base is None:
            self.save_sample_base = save_sample_base

        self.saved_model = saved_model
        self.save_generated_strokes = save_generated_strokes
        if not save_generated_stroke_base is None:
            self.gen_stroke_base = save_generated_stroke_base

        cuda_msg = "Using CUDA" if use_cuda else "Not using CUDA"
        print(cuda_msg)
        log.info(cuda_msg)

    def train(self, train_strokeset, model_save_loc = None, hidden_size = 256, n_gaussians = 20, Kmixtures = 10, dropout = .2):
        """train

           A model is specified, and the trained model is loaded if one is available.  If not,
           then the model is trained, and the resulting model is saved for future use.
        """
        model = None

        model_file = model_save_loc
        if model_file is None:
            model_file = self.saved_model

        if not model_file is None:
            model = self.load(model_file)

        if model is None:
            torch.cuda.empty_cache()
            model = HandwritingSynthesisModel(hidden_size, n_gaussians, Kmixtures, dropout)

            log.info(f"Training network...")
            model.eval()
            model = self.train_network(model, train_strokeset, model_file, epochs = self.epoch, generate = True)

            self.trained_model = model

    def load(self, model_save_loc = None):
        """load

           Load a trained model is loaded if one is available. Either the loaded model or
           None is returned.
        """
        if not self.trained_model is None:
            return  # Model is already loaded

        torch.cuda.empty_cache()
        model = HandwritingSynthesisModel()

        model_file = model_save_loc
        if model_file is None:
            model_file = self.saved_model

        if os.path.exists(model_file):  # Load model if previously saved
            load_msg = f"Loading previously saved model file: {model_file}"
            log.info(load_msg)
            print(load_msg)
            try:
                if not use_cuda:
                    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
                else:
                    model.load_state_dict(torch.load(model_file))

                self.trained_model = model
                return model
            except:
                msg = f"Error loading saved model {model_file}"
                log.error(msg, exc_info=True)
                print(msg)
                return None
        else:
            log.info(f"Could not load saved model file {model_file}")
            return None

    def get_bounds(self, data, factor):
        """get_bounds

           Compute the bounds needed to draw the strokes of the sample.
        """
        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0

        abs_x = 0
        abs_y = 0
        for i in range(len(data)):
            x = float(data[i, 0]) / factor
            y = float(data[i, 1]) / factor
            abs_x += x
            abs_y += y
            min_x = min(min_x, abs_x)
            min_y = min(min_y, abs_y)
            max_x = max(max_x, abs_x)
            max_y = max(max_y, abs_y)

        return (min_x, max_x, min_y, max_y)

    def get_stroke_path(self, data, factor = 10, offset_x = 0, offset_y = 0):
        """get_stroke_path

           Get the path of a stroke sample in the SVG API format for 
           given data.
        """
        min_x, max_x, min_y, max_y = self.get_bounds(data, factor)

        lift_pen = 1

        abs_x = offset_x + 25 - min_x
        abs_y = offset_y + 25 - min_y
        p = "M%s,%s " % (abs_x, abs_y)

        command = "m"

        for i in range(len(data)):
            if (lift_pen == 1):
                command = "m"
            elif (command != "l"):
                command = "l"
            else:
                command = ""
            x = float(data[i, 0]) / factor
            y = float(data[i, 1]) / factor
            lift_pen = data[i, 2]
            p += command + str(x) + "," + str(y) + " "

        return p

    def save_generated_stroke(self, data, factor=10, show_save_loc = False):
        """save_generated_stroke

           Using the array format of the stroke data, draw the handwriting sample
           and optionally save it to a file for viewing.

        """
        # File management
        if self.save_generated_strokes:
            gen_stroke_save_name = self.gen_stroke_base + r"_samples_" + str(self.gen_stroke_num) + r".svg"
            os.makedirs(os.path.dirname(gen_stroke_save_name), exist_ok=True)
        else:
            return  # If we aren't saving the strokes, then don't bother with the rest

        min_x, max_x, min_y, max_y = self.get_bounds(data, factor)
        dims = (50 + max_x - min_x, 50 + max_y - min_y)

        dwg = svgwrite.Drawing(gen_stroke_save_name, size=dims)
        dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

        the_color = "black"
        stroke_width = 1

        dwg.add(dwg.path(self.get_stroke_path(data, factor)).stroke(the_color, stroke_width).fill("none"))

        if self.save_generated_strokes:
            msg = f"Saving generated stroke in {gen_stroke_save_name}"
            log.info(msg)
            if show_save_loc:
                print(msg)

            try:
                dwg.save()
                self.gen_stroke_num += 1
            except:
                log.error(f"Could not save stroke drawing", exc_info=True)

    def save_generated_stroke_biases(self, strokes, factor=10, show_save_loc = False, biases = [0., .1, .5, 2, 5, 10]):
        """save_generated_stroke_biases

           Using the array format of the stroke data, draw the handwriting sample
           and optionally save it to a file for viewing.  In this variation of
           stroke generation, in place of a single sample, multiple samples
           are generated using all the biases in the parameter list for
           easy visual comparison.
        """
        # File management
        if self.save_generated_strokes:
            gen_stroke_save_name = self.gen_stroke_base + r"_sample_with_biases_" + str(self.gen_stroke_num) + r".svg"
            os.makedirs(os.path.dirname(gen_stroke_save_name), exist_ok=True)
        else:
            return  # If we aren't saving the strokes, then don't bother with the rest

        # Find total size of canvas that the drawing will need
        tot_x = 0
        tot_y = 0
        for stroke in strokes:
            min_x, max_x, min_y, max_y = self.get_bounds(stroke, factor)
            tot_x += 50 + max_x - min_x
            tot_y += 50 + max_y - min_y

        dims_total = (tot_x, tot_y)

        # Create the drawing cavas
        dwg = svgwrite.Drawing(gen_stroke_save_name, size=dims_total)
        dwg.add(dwg.rect(insert=(0, 0), size=dims_total, fill='white'))

        the_color = "black"
        stroke_width = 1

        # Draw the strokes and their biases
        offset_x = 50   # Offset x for stroke
        offset_y = 0    # Offset y for stroke
        text_x = 10     # Starting x point for text
        text_y = 0      # Starting y point for text
        for i in range(len(biases)):
            # Get dimensions for current stroke
            stroke = strokes[i]
            min_x, max_x, min_y, max_y = self.get_bounds(stroke, factor)
            dims = (50 + max_x - min_x, 50 + max_y - min_y)

            # Write the bias size
            bias = biases[i]
            text_y = offset_y + dims[1]/2  # y point for text
            dwg.add(dwg.text(f"{bias}", insert=(text_x, text_y), font_size="15px"))

            # Draw current stroke
            dwg.add(dwg.path(self.get_stroke_path(stroke, factor, offset_x, offset_y)).stroke(the_color, stroke_width).fill("none"))
            
            # Update offsets
            offset_x += 0        # Align to the left
            offset_y += dims[1]  # Down to next row

        if self.save_generated_strokes:
            msg = f"Saving generated stroke in {gen_stroke_save_name}"
            log.info(msg)
            if show_save_loc:
                print(msg)

            try:
                dwg.save()
                self.gen_stroke_num += 1
            except:
                log.error(f"Could not save stroke drawing", exc_info=True)

    def line_plot(self, strokes, title):
        """line_plot

           Render a single handwriting sample.  Note that the handwriting samples will appear cropped.
           That is because the trainer only uses the first *sequence_length* stroke points for training
           and therefore for plotting here.
        """

        # Skip check
        if not self.save_samples and not self.display_images:
            return  # Do nothing if user doesn't want the samples

        if is_linux:
            matplotlib.use('Agg')  # Non-interactive backend

        # File management
        if self.save_samples:
            hw_save_name = self.save_sample_base + r"_samples_" + str(self.hw_num) + r".png"
            log.info(f"Saving: {hw_save_name}")
            try:
                os.makedirs(os.path.dirname(hw_save_name), exist_ok=True)
            except:
                log.error(f"Could not create plot file", exc_info=True)

        fig = plt.figure(figsize=(20,2))
        eos_preds = np.where(strokes[:,-1] == 1)
        eos_preds = [0] + list(eos_preds[0]) + [-1] # Add start and end indices

        for i in range(len(eos_preds)-1):
            start = eos_preds[i]+1
            stop = eos_preds[i+1]
            plt.plot(strokes[start:stop,0], strokes[start:stop,1],'b-', linewidth=2.0)
        plt.title(title)
        plt.gca().invert_yaxis()

        if self.save_samples:
            log.info(f"Saving handwriting sample \"{title}\" in {hw_save_name}")
            try:
                fig.savefig(hw_save_name)
                self.hw_num += 1
            except:
                log.error(f"Could not save handwriting sample to {hw_save_name}", exc_info=True)

        if self.display_images:
            plt.show()

        plt.close('all')

    def show_hw_samples(self, x, s):
        """show_hw_samples

           Convenience method to show or save all the samples in a batch.
        """
        save_display_flag = self.display_images

        for i in range(self.n_batch):
            r = x[i]
            strokes = r.copy()
            strokes[:,:-1] = np.cumsum(r[:,:-1], axis=0)
            self.line_plot(strokes, s[i])

            if self.display_images:   # Only display the first in a batch so user isn't flooded
                self.display_images = False

        self.display_images = save_display_flag

    def one_hot(self, s):
        """one_hot

           Transforms a string sequence into a one-hot matrix. Dimensions of the output 
           one-hot matrix are (string length, len(alphabet)).

        """
        # Index position 0 means "unknown"
        alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

        if s is None:
            seq = [0]
            log.debug(f"One hotting: nothing")
        else:
            s += ' '
            seq = [alphabet.find(char) + 1 for char in s]
            log.debug(f"One hotting: {s}")

        one_hot = np.zeros((len(s),len(alphabet)+1))
        one_hot[np.arange(len(s)),seq] = 1
        return one_hot

    def plot_heatmaps(self, text, Phis, Ws):
        """plot_heatmaps

           Plots Phis and soft-window heatmaps. It corresponds to the values of equations 
           46 and 47 of the paper. 


           46)
                        K
                      =====       /              2\
                      \      k    |/  k\ / k    \ |
            Φ(t,u) =   >    α  exp||-β | |κ  - u| |
                      /      t    \\  t/ \ t    / /
                      =====
                      k = 1
            47)
                   U
                 =====
                 \
            w  =  >    Φ(t,u) c
             t   /             u
                 =====
                 u = 1

        """

        # File management
        plot_save_name = self.save_plot_base + r"_weights_" + str(self.plot_num) + r".png"
        try:
            os.makedirs(os.path.dirname(plot_save_name), exist_ok=True)
        except:
            log.error(f"Could not initialize plot file {plot_save_name}", exc_info=True)

        if is_linux:
            matplotlib.use('Agg')  # Non-interactive backend

        # Crop data array for more visually appealing plots
        def trim_empty_space(m, tol = 1e-3, r = None, c = None):
            # Trim zeroes at the bottom of the array
            if r is None:
                for first_row_elem_with_zeroes in range(len(m[:,0])-1, 0, -1):  # Go in reverse so meaningful data gaps are retained
                    row = m[first_row_elem_with_zeroes,:]              # Examine each row in turn
                    if np.sum(row) > tol:                              # If the entire row is above the tolerance, we are done
                        break
                last_nonzero_row = first_row_elem_with_zeroes + 1
            else:
                last_nonzero_row = r                                   # Or, the caller can just specify the amount to cut

            # Trim zeroes to the right side of the array
            if c is None:
                for first_col_elem_with_zeroes in range(len(m[0,:])-1, 0, -1):  # Go in reverse so meaningful data gaps are retained
                    col = m[:,first_col_elem_with_zeroes]              # Examine each column in turn
                    if np.sum(col) > tol:                              # If the entire column is above the tolerance, we are done
                        break
                last_nonzero_col = first_col_elem_with_zeroes + 1
            else:
                last_nonzero_col = c                                   # Or, the caller can just specify the amount to cut

            return m[:last_nonzero_row, :last_nonzero_col]             # Execute the crop

        # Two subplots in figure
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (4, 8), constrained_layout = True)
        fig.suptitle(r"Window Weights", fontsize=20, fontweight='bold')

        # Phi plot
        plot_phis = trim_empty_space(Phis)
        log.debug(f"Trimmed Phi from {np.shape(Phis)} to {np.shape(plot_phis)} for plotting")

        ax1.set_title('Phis', fontsize=14)
        ax1.set_ylabel(text, fontsize=10, style='normal', fontfamily='monospace', fontweight='bold')
        ax1.imshow(plot_phis, interpolation='nearest', origin='lower', aspect='auto', cmap=cm.jet)

        # Soft weight (w_t) plot
        plot_ws = trim_empty_space(Ws, tol = 1e-5, c = int(np.shape(plot_phis)[1]))
        log.debug(f"Trimmed weights from {np.shape(Ws)} to {np.shape(plot_ws)} for plotting")

        ax2.set_title('Soft attention window', fontsize=14)
        ax2.set_xlabel(r"Time Steps", fontsize=12, style='italic')
        ax2.set_ylabel(r"One-hot Vector", fontsize=12, style='italic')
        ax2.imshow(plot_ws, interpolation='nearest', origin='lower', aspect='auto', cmap=cm.jet)

        # Save the data arrays if we need to examine them
        #np.savetxt(".\phis.csv", Phis, delimiter=",")
        #np.savetxt(".\plot_phis.csv", plot_phis, delimiter=",")
        #np.savetxt(".\ws.csv", Ws, delimiter=",")
        #np.savetxt(".\plot_ws.csv", plot_ws, delimiter=",")

        # Plot output
        try:
            log.info(f"Saving plot {plot_save_name}")
            fig.savefig(plot_save_name)
            self.plot_num += 1
        except:
            log.error(f"Could not save plot {plot_save_name}", exc_info=True)

        if self.display_images:
            plt.show()

        plt.close('all')

    def get_n_params(self, model):
        """get_n_params

           returns the number of parameters of a model
        """

        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    def gaussian_mixture(self, y, pis, mu1s, mu2s, sigma1s, sigma2s, rhos):
        """gaussianMixture

           Implement the probability density of our next point given our 
           output vector (the Gaussian mixtures parameters). In the paper, this is given 
           by equations 23-25. This will be useful when computing the loss function.

           23)
                            M				   
                          =====        					   / e      if (x   ) = 1
			              \    	__j     	 j   j   j    |   t        ( t+1)3  
            Pr(x  	|y) =  >    ||   N(x   |μ , σ , ρ )  <    
	            t+1   t   /		  t		t+1  t 	 t   t    | 1-e	  otherwise
			              =====			                   \   t
			              j = 1

where       24)
                                   1             /    -Z    \
            N(x|μ,σ,ρ) = -------------------- exp|----------|
                                       ______    |  /     2\|
                           __         /     2    \2 \1 - ρ //
                         2 || σ  σ  \/ 1 - ρ
                               1  2
with       25)
                         2            2
                /x  - μ \    /x  - μ \    2 ρ /x  - μ \ /x  - μ \
                \ 1    1/    \ 2    2/        \ 1    1/ \ 2    2/
            Z = ---------- + ---------- - -----------------------
                     2            2                σ  σ
                    σ            σ                  1  2
                     1            2

           The Bernoulli part from the paper is excluded here. It will be computed in the 
           loss function.

           Remember the forward function of our model. gaussianMixture(...) takes for 
           parameters its outputs. As such, it computes the results of equation 23 of 
           the whole sequence over the different batches. A note on parameter y. It is 
           basically the same tensor as x but shifted one time step. Think of it as  
           𝑥𝑡+1 in equation 23. It allows the last point of a sequence to still be 
           learned correctly.
        """

        n_mixtures = pis.size(2)
    
        # Takes x1 and repeats it over the number of gaussian mixtures
        x1 = y[:,:, 0].repeat(n_mixtures, 1, 1).permute(1, 2, 0) 
        #log.debug(f"x1 shape {x1.shape}") # -> torch.Size([self.sequence_length, batch, self.n_gaussians])
    
        # First term of Z (eq 25)
        x1norm = ((x1 - mu1s) ** 2) / (sigma1s ** 2 )
        #log.debug(f"x1norm shape {x1.shape}") # -> torch.Size([self.sequence_length, batch, self.n_gaussians])
    
        x2 = y[:,:, 1].repeat(n_mixtures, 1, 1).permute(1, 2, 0)  
        #log.debug(f"x2 shape {x2.shape}") # -> torch.Size([self.sequence_length, batch, self.n_gaussians])
    
        # Second term of Z (eq 25)
        x2norm = ((x2 - mu2s) ** 2) / (sigma2s ** 2 )
        #log.debug(f"x2norm shape {x2.shape}") # -> torch.Size([self.sequence_length, batch, self.n_gaussians])
    
        # Third term of Z (eq 25)
        coxnorm = 2 * rhos * (x1 - mu1s) * (x2 - mu2s) / (sigma1s * sigma2s) 
    
        # Computing Z (eq 25)
        Z = x1norm + x2norm - coxnorm
    
        # Gaussian bivariate (eq 24)
        N = torch.exp(-Z / (2 * (1 - rhos ** 2))) / (2 * np.pi * sigma1s * sigma2s * (1 - rhos ** 2) ** 0.5) 
        #log.debug(f"N shape {N.shape}") # -> torch.Size([self.sequence_length, batch, self.n_gaussians]) 
    
        # Pr is the result of eq 23 without the eos part
        Pr = pis * N 
        #log.debug(f"Pr shape {Pr.shape}") # -> torch.Size([self.sequence_length, batch, self.n_gaussians])   
        Pr = torch.sum(Pr, dim=2) 
        #log.debug(f"Pr shape {Pr.shape}") # -> torch.Size([self.sequence_length, batch])   
    
        if use_cuda:
            Pr = Pr.cuda()
    
        return Pr

    def loss_fn(self, Pr, y, es):
        """loss_fn

           The goal is to maximize the likelihood of our estimated bivariate normal 
           distributions and Bernoulli distribution. We generate parameters for our 
           distributions but we want them to fit as best as possible to the data. Each 
           training step's goal is to converge toward the best parameters for the data. 

           In the paper, the loss is given by equation 26:

           26)
                    T
	              =====		===== 				         /
                  \		    \     __j        j  j  j     | log e     if (x   ) = 1
            L(x) = >   -log  >    || N(x   |μ ,σ ,ρ ) - <       t       ( t+1)3
	              /	       	/	    t   t+1	 t  t  t     | log(1-e ) otherwise
	              =====		=====                        \        t
	              t = 1		  j

           We previously calculated the first element of the equation in 
           gaussianMixture(...). What's left is to add the Bernoulli loss (second part 
           of our equation). The loss of each time step is summed up and averaged over 
           the batches.
        """

        loss1 = - torch.log(Pr + self.eps) # -> torch.Size([self.sequence_length, batch])    
        bernoulli = torch.zeros_like(es) # -> torch.Size([self.sequence_length, batch])
    
        bernoulli = y[:, :, 2] * es + (1 - y[:, :, 2]) * (1 - es)
    
        loss2 = - torch.log(bernoulli + self.eps)
        loss = loss1 + loss2 
        log.debug(f"loss shape {loss.shape}") # -> torch.Size([self.sequence_length, batch])  
        loss = torch.sum(loss, 0) 
        log.debug(f"loss shape {loss.shape}") # -> torch.Size([batch]) 
    
        return torch.mean(loss);

    def train_network(self, model, train_strokeset, model_save_loc, epochs = 5, generate = True):
        """train_network

           This uses an Adam optimizer with a learning rate of 0.005. The gradients are clipped 
           inside [-gradient_threshold, gradient_treshold] to avoid exploding gradient.

           The model is saved after each epoch.  This not only makes the model available
           later without the need to retrain; it also guards against unexpected interruptions
           of the training process so that we don't have to start over completely.

           As the training proceeds, information of interest is saved, such as the loss
           for epoch and batch.  This information is then plotted at the end of the
           training.  In addition, heat maps like those in the paper are plotted every
           100 batches.

           The heart of the whole algorithm happens when we call forward() on the custom
           LSTM model, which returns the information that we need to calculate the
           Gaussian mixture probabilities.  Once computed, that probability is then fed
           into the loss function.  These 3 steps are repeated for every batch/epoch.
        """

        if is_linux:
            matplotlib.use('Agg')  # Non-interactive back-end

        data_loader = LSTMDataInterface(train_strokeset, self.n_batch, self.sequence_length, 20, U_items=self.U_items) # 20 = datascale
    
        optimizer = optim.Adam(model.parameters(), lr=0.005)
    
        # A sequence the model is going to try to write as it learns
        c0 = np.float32(self.one_hot("Handwriting sample"))
        c0 = torch.from_numpy(c0) 
        c0 = torch.unsqueeze(c0, 0) # torch.Size(self.n_batch, self.U_items, len(alphabet))
        start = time.time()
    
        if use_cuda:
            model = model.cuda()
            c0 = c0.cuda()
        
        # Arrays to plot loss over time
        time_batch = []
        time_epoch = [0]
        loss_batch = []
        loss_epoch = []
    
        # Loop over epochs
        epoch_msg = f"Training over {epochs} epochs"
        print(epoch_msg)
        log.info(epoch_msg)

        for epoch in range(epochs):
            log.info(f"Processing epoch {epoch}")
            data_loader.reset_batch_pointer()
        
            # Loop over batches
            for batch in range(data_loader.num_batches):
                # Loading a batch (x : stroke sequences, y : same as x but shifted 1 timestep, c : one-hot encoded character sequence ofx)
                log.info(f"Processing batch {batch}")
                x, y, s, c = data_loader.next_batch()
                self.show_hw_samples(x, s)

                x = np.float32(np.array(x)) # -> (self.n_batch, self.sequence_length, 3)
                y = np.float32(np.array(y)) # -> (self.n_batch, self.sequence_length, 3)
                c = np.float32(np.array(c))

                x = torch.from_numpy(x).permute(1, 0, 2) # torch.Size([self.sequence_length, self.n_batch, 3])
                y = torch.from_numpy(y).permute(1, 0, 2) # torch.Size([self.sequence_length, self.n_batch, 3])
                c = torch.from_numpy(c) # torch.Size(self.n_batch, self.U_items, len(alphabet))
            
                if use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                    c = c.cuda()
            
                # Forward pass
                es, pis, mu1s, mu2s, sigma1s, sigma2s, rhos = model.forward(x, c)
            
                # Calculate probability density and loss
                Pr = self.gaussian_mixture(y, pis, mu1s, mu2s, sigma1s, sigma2s, rhos)
                loss = self.loss_fn(Pr,y, es)
                log.debug(f"Pr = {Pr}, Loss = {loss}, Es = {es}, Pi_s = {pis}, Mu1_s = {mu1s}, Mu2_s = {mu2s}, Sigma1_s = {sigma1s}, Sigma2_s = {sigma2s}, Rho_s = {rhos}")
            
                # Back propagation
                optimizer.zero_grad()
                loss.backward()
            
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_threshold)
                optimizer.step()
            
                # Useful infos over training
                if batch % 10 == 0:
                    epoch_msg = f"Epoch : {epoch} - step {batch}/{data_loader.num_batches} - loss {loss.item():.3f} took {(time.time() - start):.2f} seconds"
                    print(epoch_msg)
                    log.info(epoch_msg)
                    start = time.time()
                
                    # Plot heatmaps every 100 batches
                    if batch % 100 == 0:
                        self.plot_heatmaps(s[0], model.Phis.transpose(0, 1).detach().numpy(), model.Ws.transpose(0, 1).detach().numpy())
                    
                    # Generate a sequence every 500 batches to watch as the training progresses   
                    if generate and batch % 500 == 0 :
                        x0 = torch.Tensor([0,0,1]).view(1,1,3)

                        if use_cuda:
                            x0 = x0.cuda()
                    
                        for i in range(5):
                            sequence = model.generate_sequence(x0, c0, bias = 10)
                            seqMsg = f"Sequence shape = {sequence.shape}"
                            log.debug(seqMsg)
                            self.save_generated_stroke(sequence, factor=0.5)
                        print()   # Line return
                    
                # Save loss per batch
                time_batch.append(epoch + batch / data_loader.num_batches)
                loss_batch.append(loss.item())
        
            # Save loss per epoch
            time_epoch.append(epoch + 1)
            loss_epoch.append(sum(loss_batch[epoch * data_loader.num_batches : (epoch + 1)*data_loader.num_batches-1]) / data_loader.num_batches)
        
            # Save model after each epoch
            log.info(f"Saving model after epoch {epoch} in {model_save_loc}")
            os.makedirs(os.path.dirname(model_save_loc), exist_ok=True)
            torch.save(model.state_dict(), model_save_loc)
        
        # Plot loss 
        plt.plot(time_batch, loss_batch)
        plt.plot(time_epoch, [loss_batch[0]] + loss_epoch, color="orange", linewidth=5)
        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.show()
        

        return model

    def as_handwriting(self, text, bias = 10, show_biases = False):
        """as_handwriting

           Generate handwriting of a text string.  Input a maximum of 80 characters.
        """

        assert len(text) <= 80   # Restrict the length for performance; don't want to gen from a book!
        assert not self.trained_model is None  # Must have a trained model first to work!!

        msg = f"Generating handwriting from text \"{text}\""
        print(msg)
        log.info(msg)

        # c is the text to generate
        c0 = np.float32(self.one_hot(text))
        c0 = torch.from_numpy(c0) 
        c0 = torch.unsqueeze(c0, 0)

        # Starting sample (initially empty)
        x0 = torch.Tensor([0,0,1]).view(1,1,3)

        if use_cuda:
            x0 = x0.cuda()

        # Ask the trained model to generate the stroke sequence
        if show_biases:
            biases = [0., .1, .5, 2, 5, 10]
            sequences = []

            for bias in biases:
                sequences.append(self.trained_model.generate_sequence(x0, c0, bias))
                #print()
            
            self.save_generated_stroke_biases(sequences, factor = 0.5, biases = biases)
        else:
            sequence = self.trained_model.generate_sequence(x0, c0, bias)
            #print()
            seq_msg = f"Sequence shape for text {text} = {sequence.shape}"
            log.debug(seq_msg)
            self.save_generated_stroke(sequence, factor=0.5, show_save_loc = True)

    def smooth_handwriting(self, sample, bias = 10, show_biases = False):
        """smooth_handwriting
        "A MOVE to stop Mr. Gaitskell"
           Use the specified sample file to smooth it.  The sample must be in the IAM 
           online data format (XML) at this time.  The result will be saved to an SVG 
           file using the path specified in the *generated-save* flag.

           Technically, the stroke data is read in from the sample file and converted
           to the synthesis model format using the LSTMDataInterface.  Then, that array
           is passed into the sequence generator to 'prime' it as described in section
           5.5. of the paper.
        """

        assert not self.trained_model is None  # Must have a trained model first to work!!

        sample = LSTMDataInterface(sample)
        msg = f"Smoothing for text \"{sample.ascii_data[0]}\""
        print(msg)
        log.info(msg)

        # c is the text to generate
        text = 2*sample.ascii_data[0]
        c0 = np.float32(self.one_hot(text))
        c0 = torch.from_numpy(c0) 
        c0 = torch.unsqueeze(c0, 0)

        # Starting sample
        x0 = torch.Tensor([0,0,1]).view(1,1,3)
        prime0 = self.build_priming_sequence(x0, sample.stroke_data[0])

        if use_cuda:
            x0 = x0.cuda()
            prime0 = prime0.cuda()

        # Ask the trained model to generate the stroke sequence
        if show_biases:
            biases = [0., .1, .5, 2, 5, 10]
            sequences = []

            for bias in biases:
                sequences.append(self.trained_model.generate_sequence(prime0, c0, bias))
                #print()
            
            self.save_generated_stroke_biases(sequences, factor = 0.5, biases = biases)
        else:
            sequence = self.trained_model.generate_sequence(prime0, c0, bias)
            print()
            seq_msg = f"Sequence shape for text smoothing = {sequence.shape}"
            log.debug(seq_msg)
            self.save_generated_stroke(sequence, factor=0.5, show_save_loc = True)

    def build_priming_sequence(self, x0, data):
        sequence = x0
        sample = x0

        for i in range(len(data)):
            sample = torch.zeros_like(x0) # torch.Size([1, 1, 3])
            d = torch.from_numpy(data[i])
            sample[0, 0, 0] = d[0]
            sample[0, 0, 1] = d[1]
            sample[0, 0, 2] = d[2]
            
            sequence = torch.cat((sequence, sample), 0) # torch.Size([sequence_length, 1, 3])

        sequence = sequence[1:]     # Remove first row
        return sequence