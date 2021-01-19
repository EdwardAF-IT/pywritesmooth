# Basics
import os, time, svgwrite, numpy as np, logging as log

# Project
from .TrainerInterface import TrainerInterface
from .HandwritingSynthesisModel import HandwritingSynthesisModel
from pywritesmooth.Data.LSTMDataInterface import LSTMDataInterface

# Neural Networks
import torch
from torch import nn, optim
use_cuda = False
use_cuda = torch.cuda.is_available()

# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import display, SVG

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
       HandwritingSynthesisModel).  The loss is computed by following the Guassian mixture
       equations from the paper.

       Finally, there are methods to optionally create plots of the sample strokes and
       heatmaps from the paper.

       Adapted from: https://github.com/adeboissiere/Handwriting-Prediction-and-Synthesis
    """
    def init(self):
        self.n_batch = 20
        self.sequence_length = 400
        self.U_items = int(self.sequence_length/25)
        self.eps = float(np.finfo(np.float32).eps)

        self.hidden_size = 256
        self.n_layers = 3
        self.n_gaussians = 20
        self.Kmixtures = 10

        # Hyperparameters
        self.gradient_threshold = 10
        self.dropout = 0.2

    def __init__(self, saved_model = None):
        log.debug("In ltsm con")
        self.init()
        self.saved_model = saved_model

        cudaMsg = "Using CUDA" if use_cuda else "Not using CUDA"
        print(cudaMsg)
        log.info(cudaMsg)

    def train(self, trainStrokeset, modelSaveLoc = None, hidden_size = 256, n_gaussians = 20, Kmixtures = 10, dropout = .2):
        """train

           A model is specified, and the trained model is loaded if one is available.  If not,
           then the model is trained, and the resulting model is saved for future use.
        """
        torch.cuda.empty_cache()
        model = HandwritingSynthesisModel(hidden_size, n_gaussians, Kmixtures, dropout)

        modelFile = modelSaveLoc
        if modelFile is None:
            modelFile = self.saved_model

        if os.path.exists(modelFile):  # Load model if previously saved
            log.info(f"Loading model: {modelFile}")
            model.load_state_dict(torch.load(modelFile))
        else:
            log.info(f"Training network...")
            model.eval()
            model = self.train_network(model, trainStrokeset, modelFile, epochs = 2, generate = True)

        self.trained_model = model

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

    def draw_strokes(self, data, factor=10, svg_filename='sample.svg'):
        """draw_strokes

           Using the array format of the stroke data, draw the handwriting sample
           and optionally save it to a file for viewing.

        """
        min_x, max_x, min_y, max_y = self.get_bounds(data, factor)
        dims = (50 + max_x - min_x, 50 + max_y - min_y)

        dwg = svgwrite.Drawing(svg_filename, size=dims)
        dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

        lift_pen = 1

        abs_x = 25 - min_x
        abs_y = 25 - min_y
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

        the_color = "black"
        stroke_width = 1

        dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))

        dwg.save()
        display(SVG(dwg.tostring()))

    def line_plot(self, strokes, title):
        plt.figure(figsize=(20,2))
        eos_preds = np.where(strokes[:,-1] == 1)
        eos_preds = [0] + list(eos_preds[0]) + [-1] # Add start and end indices
        for i in range(len(eos_preds)-1):
            start = eos_preds[i]+1
            stop = eos_preds[i+1]
            plt.plot(strokes[start:stop,0], strokes[start:stop,1],'b-', linewidth=2.0)
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def one_hot(self, s):
        """one_hot

           Transforms a string sequence into a one-hot matrix. Dimensions of the output 
           one-hot matrix are (string length, len(alphabet)).

        """
        # Index position 0 means "unknown"
        alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        seq = [alphabet.find(char) + 1 for char in s]

        one_hot = np.zeros((len(s),len(alphabet)+1))
        one_hot[np.arange(len(s)),seq] = 1
        return one_hot

    def plot_heatmaps(self, Phis, Ws):
        """plot_heatmaps

           plots Phis and soft-window heatmaps. It corresponds to the values of equations 
           46 and 47 of the paper. 
        """

        fig = plt.figure(figsize=(16,4))
        plt.subplot(121)
        plt.title('Phis', fontsize=20)
        plt.xlabel("Time stself.eps", fontsize=15)
        plt.ylabel("Ascii #", fontsize=15)
    
        plt.imshow(Phis, interpolation='nearest', aspect='auto', cmap=cm.jet)
        plt.subplot(122)
        plt.title('Soft attention window', fontsize=20)
        plt.xlabel("Time stself.eps", fontsize=15)
        plt.ylabel("One-hot vector", fontsize=15)
        plt.imsave('test.png', Ws, cmap=cm.jet)

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

    def gaussianMixture(self, y, pis, mu1s, mu2s, sigma1s, sigma2s, rhos):
        """gaussianMixture

           Implement the probability density of our next point given our 
           output vector (the Gaussian mixtures parameters). In the paper, this is given 
           by equations 23-25. This will be useful when computing the loss function.

           The Bernouilli part from the paper is excluded here. It will be computed in the 
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
        log.debug(f"x1 shape {x1.shape}") # -> torch.Size([self.sequence_length, batch, self.n_gaussians])
    
        # First term of Z (eq 25)
        x1norm = ((x1 - mu1s) ** 2) / (sigma1s ** 2 )
        log.debug(f"x1norm shape {x1.shape}") # -> torch.Size([self.sequence_length, batch, self.n_gaussians])
    
        x2 = y[:,:, 1].repeat(n_mixtures, 1, 1).permute(1, 2, 0)  
        log.debug(f"x2 shape {x2.shape}") # -> torch.Size([self.sequence_length, batch, self.n_gaussians])
    
        # Second term of Z (eq 25)
        x2norm = ((x2 - mu2s) ** 2) / (sigma2s ** 2 )
        log.debug(f"x2norm shape {x2.shape}") # -> torch.Size([self.sequence_length, batch, self.n_gaussians])
    
        # Third term of Z (eq 25)
        coxnorm = 2 * rhos * (x1 - mu1s) * (x2 - mu2s) / (sigma1s * sigma2s) 
    
        # Computing Z (eq 25)
        Z = x1norm + x2norm - coxnorm
    
        # Gaussian bivariate (eq 24)
        N = torch.exp(-Z / (2 * (1 - rhos ** 2))) / (2 * np.pi * sigma1s * sigma2s * (1 - rhos ** 2) ** 0.5) 
        log.debug(f"N shape {N.shape}") # -> torch.Size([self.sequence_length, batch, self.n_gaussians]) 
    
        # Pr is the result of eq 23 without the eos part
        Pr = pis * N 
        log.debug(f"Pr shape {Pr.shape}") # -> torch.Size([self.sequence_length, batch, self.n_gaussians])   
        Pr = torch.sum(Pr, dim=2) 
        log.debug(f"Pr shape {Pr.shape}") # -> torch.Size([self.sequence_length, batch])   
    
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

           We previously calculated the first element of the equation in 
           gaussianMixture(...). What's left is to add the Bernoulli loss (second part 
           of our equation). The loss of each time step is summed up and averaged over 
           the batches.
        """

        loss1 = - torch.log(Pr + self.eps) # -> torch.Size([self.sequence_length, batch])    
        bernouilli = torch.zeros_like(es) # -> torch.Size([self.sequence_length, batch])
    
        bernouilli = y[:, :, 2] * es + (1 - y[:, :, 2]) * (1 - es)
    
        loss2 = - torch.log(bernouilli + self.eps)
        loss = loss1 + loss2 
        log.debug(f"loss shape {loss.shape}") # -> torch.Size([self.sequence_length, batch])  
        loss = torch.sum(loss, 0) 
        log.debug(f"loss shape {loss.shape}") # -> torch.Size([batch]) 
    
        return torch.mean(loss);

    def train_network(self, model, trainStrokeset, modelSaveLoc, epochs = 5, generate = True):
        """train_network

           This uses an Adam optimizer with a learning rate of 0.005. The gradients are clipped 
           inside [-gradient_threshold, gradient_treshold] to avoid exploding gradient.

           The model is saved after each epoch.  This not only makes the model available
           later without the need to retrain; it also guards against unexpected interruptions
           of the training process so that we don't have to start over completely.

           As the training proceeds, information of interest is saved, such as the loss
           for epoch and batch.  This information is then plotted at the end of the
           training.  In addition, heatmaps like those in the paper are plotted every
           100 batches.

           The heart of the whole algorithm happens when we call forward() on the custom
           LSTM model, which returns the information that we need to calculate the
           Gaussian mixture probabilities.  Once computed, that probability is then fed
           into the loss function.  These 3 steps are repeated for every batch/epoch.
        """

        data_loader = LSTMDataInterface(trainStrokeset, self.n_batch, self.sequence_length, 20, U_items=self.U_items) # 20 = datascale
    
        optimizer = optim.Adam(model.parameters(), lr=0.005)
    
        # A sequence the model is going to try to write as it learns
        c0 = np.float32(self.one_hot("writing is hard!"))
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
        for epoch in range(epochs):
            log.debug(f"Processing epoch {epoch}")
            data_loader.reset_batch_pointer()
        
            # Loop over batches
            for batch in range(data_loader.num_batches):
                # Loading a batch (x : stroke sequences, y : same as x but shifted 1 timestep, c : one-hot encoded character sequence ofx)
                log.debug(f"Processing batch {batch}")
                x, y, s, c = data_loader.next_batch()
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
                Pr = self.gaussianMixture(y, pis, mu1s, mu2s, sigma1s, sigma2s, rhos)
                loss = self.loss_fn(Pr,y, es)
                log.debug(f"Pr = {Pr}, Loss = {loss}, Es = {es}, Pi_s = {pis}, Mu1_s = {mu1s}, Mu2_s = {mu2s}, Sigma1_s = {sigma1s}, Sigma2_s = {sigma2s}, Rho_s = {rhos}")
            
                # Back propagation
                optimizer.zero_grad()
                loss.backward()
            
                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_threshold)
                optimizer.step()
            
                # Useful infos over training
                if batch % 10 == 0:
                    print("Epoch : ", epoch, " - step ", batch, "/", data_loader.num_batches, " - loss ", loss.item(), " in ", time.time() - start)
                    log.info(f"Epoch : {epoch} - step {batch}/{data_loader.num_batches} - loss {loss.item()} in {time.time() - start}")
                    start = time.time()
                
                    # Plot heatmaps every 100 batches
                    if batch % 100 == 0:
                        print(s[0])
                        self.plot_heatmaps(model.Phis.transpose(0, 1).detach().numpy(), model.Ws.transpose(0, 1).detach().numpy())
                    
                    # Generate a sequence every 500 batches     
                    if generate and batch % 500 == 0 :
                        x0 = torch.Tensor([0,0,1]).view(1,1,3)

                        if use_cuda:
                            x0 = x0.cuda()
                    
                        for i in range(5):
                            sequence = model.generate_sequence(x0, c0, bias = 10)
                            seqMsg = f"Sequence shape = {sequence.shape}"
                            print(seqMsg)
                            log.info(seqMsg)
                            self.draw_strokes(sequence, factor=0.5)
                    
                # Save loss per batch
                time_batch.append(epoch + batch / data_loader.num_batches)
                loss_batch.append(loss.item())
        
            # Save loss per epoch
            time_epoch.append(epoch + 1)
            loss_epoch.append(sum(loss_batch[epoch * data_loader.num_batches : (epoch + 1)*data_loader.num_batches-1]) / data_loader.num_batches)
        
            # Save model after each epoch
            log.info(f"Saving model after epoch {epoch} in {modelSaveLoc}")
            os.makedirs(os.path.dirname(modelSaveLoc), exist_ok=True)
            torch.save(model.state_dict(), modelSaveLoc)
        
        # Plot loss 
        plt.plot(time_batch, loss_batch)
        plt.plot(time_epoch, [loss_batch[0]] + loss_epoch, color="orange", linewidth=5)
        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.show()
        

        return model