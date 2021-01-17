# Neural Networks
import torch
import torch.nn as nn

class HandwritingSynthesisModel(nn.Module):
    """
    Adapted from: https://github.com/adeboissiere/Handwriting-Prediction-and-Synthesis
    """

    def __init__(self, hidden_size = 256, n_gaussians = 20, Kmixtures = 10, dropout = 0.2, alphabet_size = 64):
        super(HandwritingSynthesisModel, self).__init__()
        
        self.Kmixtures = Kmixtures
        self.n_gaussians = n_gaussians
        self.alphabet_size = alphabet_size
        
        self.hidden_size1 = hidden_size
        self.hidden_size2 = hidden_size
        self.hidden_size3 = hidden_size
        
        # input_size1 includes x, y, eos and len(w_t_1) given by alphabet_size (see eq 52)
        self.input_size1 = 3 + alphabet_size
        
        # input_size2 includes x, y, eos, len(w_t) given by alphabet_size (see eq 47) and hidden_size1
        self.input_size2 = 3 + alphabet_size + self.hidden_size1
        
        # input_size3 includes x, y, eos, len(w_t) given by alphabet_size (see eq 47) and hidden_size2
        self.input_size3 = 3 + alphabet_size + self.hidden_size2
        
        # See eq 52-53 to understand the input_sizes
        self.lstm1 = nn.LSTMCell(input_size= self.input_size1 , hidden_size = self.hidden_size1)
        self.lstm2 = nn.LSTMCell(input_size= self.input_size2 , hidden_size = self.hidden_size2)
        self.lstm3 = nn.LSTMCell(input_size= self.input_size3 , hidden_size = self.hidden_size3)
        
        # Window layer takes hidden layer of LSTM1 as input and outputs 3 * Kmixtures vectors
        self.window_layer = nn.Linear(self.hidden_size1, 3 * Kmixtures)
        
        # For gaussian mixtures
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
        
        
    def forward(self, x, c, generate = False):
        # sequence length
        sequence_length = x.shape[0]
        
        # number of batches
        n_batch = x.shape[1]
        
        # Soft window vector w at t-1
        w_t_1 = torch.ones(n_batch, self.alphabet_size) # torch.Size([n_batch, len(alphabet)])
        
        # Hidden and cell state for LSTM1
        h1_t = torch.zeros(n_batch, self.hidden_size1) # torch.Size([n_batch, hidden_size1])
        c1_t = torch.zeros(n_batch, self.hidden_size1) # torch.Size([n_batch, hidden_size1])
        
        # Kappa at t-1
        kappa_t_1 = torch.zeros(n_batch, Kmixtures) # torch.Size([n_batch, Kmixtures])
        
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
        
        # Phis and Ws allow to plot heatmaps of phi et w over time
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
            
            # splits exp(window) into 3 tensors of torch.Size([n_batch, Kmixtures])
            # Eqs 48-51 of the paper
            alpha_t, beta_t, kappa_t = torch.chunk( torch.exp(window), 3, dim=1) 
            kappa_t = 0.1 * kappa_t + kappa_t_1
            
            # updates kappa_t_1 for next iteration
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
            Phi = torch.unsqueeze(Phi, 0) # torch.Size([1, U_items, n_batch])
            Phi = Phi.permute(2, 0, 1) # torch.Size([n_batch, 1, U_items])
            
            self.Phis[i, :] = Phi[0, 0, :] # To plot heatmaps
            
            # Computing wt 
            # Eq 47 of the paper
            w_t = torch.matmul(Phi, c) # torch.Size([n_batch, 1, len(alphabet)])
            w_t = torch.squeeze(w_t, 1) # torch.Size([n_batch, len(alphabet)])
            
            self.Ws[i, :] = w_t[0, :] # To plot heatmaps
            
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
        # print("es shape ", es.shape) # -> torch.Size([sequence_length, batch, 1])
        es = 1 / (1 + torch.exp(es))
        # print("es shape", es.shape) # -> torch.Size([sequence_length, batch, 1])

        pis = self.z_pi(out) * (1 + self.bias)
        # print("pis shape ", pis.shape) # -> torch.Size([sequence_length, batch, n_gaussians])
        pis = torch.softmax(pis, 2)
        # print(pis.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        mu1s = self.z_mu1(out) 
        mu2s = self.z_mu2(out)
        # print("mu shape :  ", mu1s.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        sigma1s = self.z_sigma1(out)
        sigma2s = self.z_sigma2(out)
        # print("sigmas shape ", sigma1s.shape) # -> torch.Size([sequence_length, batch, n_gaussians])
        sigma1s = torch.exp(sigma1s - self.bias)
        sigma2s = torch.exp(sigma2s - self.bias)
        # print(sigma1s.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        rhos = self.z_rho(out)
        rhos = torch.tanh(rhos)
        # print("rhos shape ", rhos.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        es = es.squeeze(2) 
        # print("es shape ", es.shape) # -> torch.Size([sequence_length, batch])

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
        mean = [mu1, mu2]
        cov = [[sigma1 ** 2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2 ** 2]]
        
        x = np.float32(np.random.multivariate_normal(mean, cov, 1))
        return torch.from_numpy(x)
        
        
    def generate_sequence(self, x0, c0, bias):
        sequence = x0
        sample = x0
        sequence_length = c0.shape[1] * 25
        
        print("Generating sequence ...")
        self.bias = bias
        f = FloatProgress(min=0, max=sequence_length)
        display(f)

        for i in range(sequence_length):
            es, pis, mu1s, mu2s, sigma1s, sigma2s, rhos = self.forward(sample, c0, True)
            
            # Selecting a mixture 
            pi_idx = np.random.choice(range(self.n_gaussians), p=pis[-1, 0, :].detach().cpu().numpy())
            
            # taking last parameters from sequence corresponding to chosen gaussian
            mu1 = mu1s[-1, :, pi_idx].item()
            mu2 = mu2s[-1, :, pi_idx].item()
            sigma1 = sigma1s[-1, :, pi_idx].item()
            sigma2 = sigma2s[-1, :, pi_idx].item()
            rho = rhos[-1, :, pi_idx].item()
            
            prediction = self.generate_sample(mu1, mu2, sigma1, sigma2, rho)
            eos = torch.distributions.bernoulli.Bernoulli(torch.tensor([es[-1, :].item()])).sample()
            
            sample = torch.zeros_like(x0) # torch.Size([1, 1, 3])
            sample[0, 0, 0] = prediction[0, 0]
            sample[0, 0, 1] = prediction[0, 1]
            sample[0, 0, 2] = eos
            
            sequence = torch.cat((sequence, sample), 0) # torch.Size([sequence_length, 1, 3])
            
            f.value += 1
        
        self.bias = 0
        self.LSTMstates = None
        
        return sequence.squeeze(1).detach().cpu().numpy()
