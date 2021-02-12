# Basics
import platform, copy, numpy as np, logging as log

# Neural Networks
import torch
import torch.nn as nn
use_cuda = False
#use_cuda = torch.cuda.is_available()

# Display
from IPython.display import SVG, display
import pywritesmooth.Utility.StrokeHelper as sh

is_linux = True if platform.system().lower() == 'linux' else False

class HandwritingSynthesisModel(nn.Module):
    """HandwritingSynthesisModel

        The network consists of LSTM cells stacked on top of each other and followed by 
        a Gaussian mixture layer with an attention mechanism in between. This network 
        includes skip connections like the paper. It is almost the same network as the 
        one in my other notebook (Handwriting prediction - Model 2) but with the attention 
        mechanism.

        The attention mechanism is implemented between LSTM1 and LSTM2. LSTM1 takes as inputs 
        the window vectors of the previous time step as well as current stroke coordinates. A 
        dense layer is used taking the output of LSTM1 to compute the parameters of the window 
        vectors. The current window vector is passed on to LSTM2 and LSTM3 as well as the stroke 
        coordinates via skip connections. LSTM2 and LSTM3 of course take the hidden vectors of 
        the LSTM1 and LSTM2 respectively. This is summarized by equations 52 and 53.


        52)
         1     /                 1                    1\
        h  = H |W    x  + W     h      + W    w    + b |
         t     \ ih1  t    h1h1  t-1      wh1  t-1    h/

         53)
          2     / n        n - 1  n  n - 1    n  n  n        n        n\
         h  = H |W   x  + W      h  h      + W  h  h      + W   w  + b |
          t     \ ih  t    h         t        h     t - 1    wh  t    h/


        The Gaussian mixtures are created using a dense layer. It takes the output of the last 
        LSTM layer. Say the hidden size is 256 and you want 10 mixtures, this allows to scale 
        your vector to the desired size. This gives ŷ of equation 17 of the paper.


        17)
                   N
                 =====
                 \          n
        ŷ = b  +  >    W   h
             y   /      h y t
                 =====   n
                 n = 1

        ŷ is then broken down into the different parameters of the mixture.
            * ê is the probability of the end of a stroke given by a Bernoulli distribution
            * w (or Π) is the weight of each Normal distribution
            * 𝜇,𝜎,𝜌 are the mean, standard deviation and correlation factor of each bivariate 
             Normal Distribution. The constructor just lays out the blocks but does not create 
             relations between them. That's the job of the forward function.

        The network is summarized by figure 12 in the paper (it does show the third hidden 
        layer which does pretty much the same thing as the second).

        Adapted from: https://github.com/adeboissiere/Handwriting-Prediction-and-Synthesis
    """

    def __init__(self, hidden_size = 256, n_gaussians = 20, Kmixtures = 10, dropout = 0.2, alphabet_size = 64):
        """__init___

           This is the constructor.  It takes the different parameters to create the different 
           blocks of the model.

               * hidden_size is the size of the output of each LSTM cell
               * n_gaussians is the number of mixtures
               * Kmixtures is the number of Gaussian functions for the window vectors
               * dropout is the dropout probability. It gives the probability to skip a cell during 
                 forward propagation. It's not implemented yet.
               * alphabet_size is the number of characters in our dictionary

        """

        super(HandwritingSynthesisModel, self).__init__()
        
        self.helper = sh.StrokeHelper
        self.EOS = False

        self.Kmixtures = Kmixtures
        self.n_gaussians = n_gaussians
        self.alphabet_size = alphabet_size
        
        self.hidden_size1 = hidden_size
        self.hidden_size2 = hidden_size
        self.hidden_size3 = hidden_size
        
        # Input_size1 includes x, y, eos and len(w_t_1) given by alphabet_size (see eq 52)
        self.input_size1 = 3 + alphabet_size
        
        # Input_size2 includes x, y, eos, len(w_t) given by alphabet_size (see eq 47) and hidden_size1
        self.input_size2 = 3 + alphabet_size + self.hidden_size1
        
        # Input_size3 includes x, y, eos, len(w_t) given by alphabet_size (see eq 47) and hidden_size2
        self.input_size3 = 3 + alphabet_size + self.hidden_size2
        
        # See eq 52-53 to understand the input_sizes
        self.lstm1 = nn.LSTMCell(input_size= self.input_size1 , hidden_size = self.hidden_size1)
        self.lstm2 = nn.LSTMCell(input_size= self.input_size2 , hidden_size = self.hidden_size2)
        self.lstm3 = nn.LSTMCell(input_size= self.input_size3 , hidden_size = self.hidden_size3)
        
        # Window layer takes hidden layer of LSTM1 as input and outputs 3 * Kmixtures vectors
        self.window_layer = nn.Linear(self.hidden_size1, 3 * Kmixtures)
        
        # For Gaussian mixtures
        self.z_e = nn.Linear(hidden_size, 1)
        self.z_pi = nn.Linear(hidden_size, n_gaussians)
        self.z_mu1 = nn.Linear(hidden_size, n_gaussians)
        self.z_mu2 = nn.Linear(hidden_size, n_gaussians)
        self.z_sigma1 = nn.Linear(hidden_size, n_gaussians)
        self.z_sigma2 = nn.Linear(hidden_size, n_gaussians)
        self.z_rho = nn.Linear(hidden_size, n_gaussians)
        
        # Bias for sampling
        self.bias = 0
        
        # Saves hidden and cell states
        self.LSTMstates = None
        self.has_been_primed = False
              
    def forward(self, x, c, generate = False):
        """forward

           This is the forward propagation. It takes x and c as inputs.

            x is a batch of stroke coordinates of sequences. Its dimensions are 
            [sequence_size, batch_size, 3]. The 3 corresponds to x and y offset of a stroke 
            and eos (= 1 when reaching an end of stroke (when the pen is raised)).

            c, a batch of one-hot encoded sentences corresponding to the stroke sequence is 
            of dimensions [n_batch, U_items, len(alphabet)]. It is estimated that a letter 
            corresponds to 25 points. U_items is the number of characters in the sequence. 
            For example, if the sequence is 400 points long, U_items = 400 / 25 = 16 characters. 
            len(alphabet) is the number of characters in our alphabet.

            Note that the forward function is also used to generate random sequences.

            The first step is to compute LSTM1. This is straightforward in PyTorch. Since the 
            LSTM cells use Pytorch, we need a for loop over the whole stroke sequence.

            After LSTM1, the code computes the attention mechanism given by equations 46-51 
            of the paper.

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

            48)
                                           1
            (αhat , βhat , κhat ) = W     h  + b
                 t      t      t       1    t    p
                                      h  p
            49)
            α  = exp/αhat \
             t      \    t/

            50)
            β  = exp/βhat \
             t      \    t/

            51)
            κ  = κ      + exp/κhat \
             t    t - 1      \    t/


            After that, the networks computes LSTM2 and LSTM3. Then it's just a matter of 
            computing 18 - 22 of the paper using a dense layer.

            18)
                      1
            e  = -----------            ==>  e  E (0,1)
             t   1 + exp/ê \                  t
                        \ t/

            19)
                                              
                         /_^j\                          =====
            __j       exp\||t/              __j         \     __j
            ||  = ----------------      ==> ||  E (0,1), >    || 
              t      M                        t         /       t
                   =====                                =====
                   \        /_^j'\                        j
                    >    exp\||  /
                   /           t
                   =====
                  j' = 1

            20)
             j    j                           j
            𝜇  = 𝜇hat                   ==>  𝜇  E R
             t    t                           t

            21)
             j      / j  \                   j
            𝜎  = exp|𝜎hat|              ==> 𝜎  > 0
             t      \ t  /                   t

            22)
             j       / j \                   j
            𝜌  = tanh|𝜌hat|             ==> 𝜌  E (-1,1)
             t       \ t  /                  t
        """

        # Sequence length
        sequence_length = x.shape[0]
        
        # Number of batches
        n_batch = x.shape[1]
        
        # Soft window vector w at t-1
        w_t_1 = torch.ones(n_batch, self.alphabet_size) # torch.Size([n_batch, len(alphabet)])
        
        # Hidden and cell state for LSTM1
        h1_t = torch.zeros(n_batch, self.hidden_size1) # torch.Size([n_batch, hidden_size1])
        c1_t = torch.zeros(n_batch, self.hidden_size1) # torch.Size([n_batch, hidden_size1])
        
        # Kappa at t-1
        kappa_t_1 = torch.zeros(n_batch, self.Kmixtures) # torch.Size([n_batch, Kmixtures])
        
        # Hidden and cell state for LSTM2
        h2_t = torch.zeros(n_batch, self.hidden_size2) # torch.Size([n_batch, hidden_size2])
        c2_t = torch.zeros(n_batch, self.hidden_size2) # torch.Size([n_batch, hidden_size2])
        
        # Hidden and cell state for LSTM3
        h3_t = torch.zeros(n_batch, self.hidden_size3) # torch.Size([n_batch, hidden_size3])
        c3_t = torch.zeros(n_batch, self.hidden_size3) # torch.Size([n_batch, hidden_size3])
        
        if generate and self.LSTMstates != None:
            h1_t = self.LSTMstates["h1_t"]
            c1_t = self.LSTMstates["c1_t"]
            h2_t = self.LSTMstates["h2_t"]
            c2_t = self.LSTMstates["c2_t"]
            h3_t = self.LSTMstates["h3_t"]
            c3_t = self.LSTMstates["c3_t"]
            w_t_1 = self.LSTMstates["w_t_1"]
            kappa_t_1 = self.LSTMstates["kappa_t_1"]
        
        out = torch.zeros(sequence_length, n_batch, self.hidden_size3)
        
        # Phis and Ws allow to plot heat maps of phi et w over time
        self.Phis = torch.zeros(sequence_length, c.shape[1])
        self.Ws = torch.zeros(sequence_length, self.alphabet_size)
        
        if use_cuda:
            w_t_1 = w_t_1.cuda()
            
            h1_t = h1_t.cuda()
            c1_t = c1_t.cuda()
            
            kappa_t_1 = kappa_t_1.cuda()
            
            h2_t = h2_t.cuda()
            c2_t = c2_t.cuda()
            
            h3_t = h3_t.cuda()
            c3_t = c3_t.cuda()
            
            out = out.cuda()
            
        for i in range(sequence_length):
            # ===== Computing 1st layer =====
            input_lstm1 = torch.cat((x[i], w_t_1), 1) # torch.Size([n_batch, input_size1])
            h1_t, c1_t = self.lstm1(input_lstm1, (h1_t, c1_t)) # torch.Size([n_batch, hidden_size1])
            
            # ===== Computing soft window =====
            window = self.window_layer(h1_t)
            
            # Splits exp(window) into 3 tensors of torch.Size([n_batch, Kmixtures])
            # Eqs 48-51 of the paper
            alpha_t, beta_t, kappa_t = torch.chunk( torch.exp(window), 3, dim=1) 
            kappa_t = 0.1 * kappa_t + kappa_t_1
            
            # Updates kappa_t_1 for next iteration
            kappa_t_1 = kappa_t
            
            u = torch.arange(0,c.shape[1], out=kappa_t.new()).view(-1,1,1) # torch.Size([U_items, 1, 1])
            
            # Computing Phi(t, u)
            # Eq 46 of the paper
            # Keep in mind the (kappa_t - u).shape is torch.Size([U_items, n_batch, Kmixtures])
            # For example :
            ## (kappa_t - u)[0, 0, :] gives kappa_t[0, :]
            ## (kappa_t - u)[1, 0, :] gives kappa_t[0, :] - 1
            ## etc
            Phi = alpha_t * torch.exp(- beta_t * (kappa_t - u) ** 2) # torch.Size([U_items, n_batch, Kmixtures])
            Phi = torch.sum(Phi, dim = 2) # torch.Size([U_items, n_batch]) 

            if self.has_been_primed and (Phi[-1][0].item() > torch.max(Phi[:-1]).item()):  # Compare the largest value in the last column of Phis to the last Phi in that column
                self.EOS = True     # This is how we know when to stop predicting stroke points

            Phi = torch.unsqueeze(Phi, 0) # torch.Size([1, U_items, n_batch])
            Phi = Phi.permute(2, 0, 1) # torch.Size([n_batch, 1, U_items])
            
            self.Phis[i, :] = Phi[0, 0, :] # To plot heat maps
            
            # Computing wt 
            # Eq 47 of the paper
            w_t = torch.matmul(Phi, c) # torch.Size([n_batch, 1, len(alphabet)])
            w_t = torch.squeeze(w_t, 1) # torch.Size([n_batch, len(alphabet)])
            
            self.Ws[i, :] = w_t[0, :] # To plot heat maps
            
            # Update w_t_1 for next iteration
            w_t_1 = w_t
            
            # ===== Computing 2nd layer =====
            input_lstm2 = torch.cat((x[i], w_t, h1_t), 1) # torch.Size([n_batch, 3 + alphabet_size + hidden_size1])
            h2_t, c2_t = self.lstm2(input_lstm2, (h2_t, c2_t)) 
            
            
            # ===== Computing 3rd layer =====
            input_lstm3 = torch.cat((x[i], w_t, h2_t), 1) # torch.Size([n_batch, 3 + alphabet_size + hidden_size2])
            h3_t, c3_t = self.lstm3(input_lstm3, (h3_t, c3_t))
            out[i, :, :] = h3_t
            
        # ===== Computing MDN =====
        es = self.z_e(out)
        #log.debug(f"es shape {es.shape}") # -> torch.Size([sequence_length, batch, 1])
        es = 1 / (1 + torch.exp(es))
        #log.debug(f"es shape {es.shape}") # -> torch.Size([sequence_length, batch, 1])

        pis = self.z_pi(out) * (1 + self.bias)
        #log.debug(f"pis shape {pis.shape}") # -> torch.Size([sequence_length, batch, n_gaussians])
        pis = torch.softmax(pis, 2)
        #log.debug(f"pis shape {pis.shape}") # -> torch.Size([sequence_length, batch, n_gaussians])

        mu1s = self.z_mu1(out) 
        mu2s = self.z_mu2(out)
        #log.debug(f"mu shape :  {mu1s.shape}") # -> torch.Size([sequence_length, batch, n_gaussians])

        sigma1s = self.z_sigma1(out)
        sigma2s = self.z_sigma2(out)
        #log.debug(f"sigmas shape {sigma1s.shape}") # -> torch.Size([sequence_length, batch, n_gaussians])
        sigma1s = torch.exp(sigma1s - self.bias)
        sigma2s = torch.exp(sigma2s - self.bias)
        #log.debug(f"sigmas shape {sigma1s.shape}") # -> torch.Size([sequence_length, batch, n_gaussians])

        rhos = self.z_rho(out)
        rhos = torch.tanh(rhos)
        #log.debug(f"rhos shape {rhos.shape}") # -> torch.Size([sequence_length, batch, n_gaussians])

        es = es.squeeze(2) 
        #log.debug(f"es shape {es.shape}") # -> torch.Size([sequence_length, batch])

        # Hidden and cell states
        if generate:
            self.LSTMstates = {"h1_t": h1_t,
                              "c1_t": c1_t,
                              "h2_t": h2_t,
                              "c2_t": c2_t,
                              "h3_t": h3_t,
                              "c3_t": c3_t,
                              "w_t_1": w_t_1,
                              "kappa_t_1": kappa_t_1}
        
        return es, pis, mu1s, mu2s, sigma1s, sigma2s, rhos
    
    def generate_sample(self, mu1, mu2, sigma1, sigma2, rho):
        """generate_sample

           Returns random coordinates based on a bivariate normal distribution given by the 
           function parameters.
        """

        mean = [mu1, mu2]
        cov = [[sigma1 ** 2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2 ** 2]]
        
        x = np.float32(np.random.multivariate_normal(mean, cov, 1))
        return torch.from_numpy(x)
        
    def generate_sequence(self, x0, c0, bias):
        """generate_sequence

           The goal of this function is to return a sequence based on either a single point or 
           beginning of sequence x0. In pseudo-code:

               * Calculate the mixture parameters of sequence x0 given one-hot encoded string c0
               * Pick a random mixture based on the weights (pi_idx)
               * Take a random point from the chosen bivariate normal distribution
               * Add it at the end of the sequence (concatenate it)
               * Repeat

           This clearly is bad practice as it has to rerun the forward prop on the entire 
           sequence each time. And the sequence gets longer and longer, which takes more 
           time to compute at each new point generated. However, this holds in just a few 
           lines and keeps the forward function cleaner.
        """

        sequence = torch.Tensor([0,0,1]).view(1,1,3)
        sample = x0
        sequence_length = 0

        self.has_been_primed = False
        
        log.info("Generating sample stroke sequence ...")
        self.bias = bias

        while not self.EOS and sequence_length < 2000:
            es, pis, mu1s, mu2s, sigma1s, sigma2s, rhos = self.forward(sample, c0, True)
            self.has_been_primed = True
            
            # Selecting a mixture 
            pi_idx = np.random.choice(range(self.n_gaussians), p=pis[-1, 0, :].detach().cpu().numpy())
            
            # Taking last parameters from sequence corresponding to chosen Gaussian
            mu1 = mu1s[-1, :, pi_idx].item()
            mu2 = mu2s[-1, :, pi_idx].item()
            sigma1 = sigma1s[-1, :, pi_idx].item()
            sigma2 = sigma2s[-1, :, pi_idx].item()
            rho = rhos[-1, :, pi_idx].item()
            
            prediction = self.generate_sample(mu1, mu2, sigma1, sigma2, rho)
            eos = torch.distributions.bernoulli.Bernoulli(torch.tensor([es[-1, :].item()])).sample()
            
            sample = torch.zeros_like(torch.Tensor([0,0,1]).view(1,1,3)) # torch.Size([1, 1, 3])
            sample[0, 0, 0] = prediction[0, 0]
            sample[0, 0, 1] = prediction[0, 1]
            sample[0, 0, 2] = eos
            
            sequence = torch.cat((sequence, sample), 0) # torch.Size([sequence_length, 1, 3])

            sequence_length += 1
            
            #self.helper.progress(count = i, total = sequence_length, status="Generating sequence      ")
        
        self.bias = 0
        self.LSTMstates = None
        self.EOS = False
        
        return sequence.squeeze(1).detach().cpu().numpy()