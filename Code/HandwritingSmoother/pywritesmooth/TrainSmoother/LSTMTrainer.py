import logging as log
from .TrainerInterface import TrainerInterface

import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format

from IPython.display import display
import os

# Sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Neural Networks
import torch
import torch.nn as nn

#import pytorch_lightning as pl
#from pytorch_lightning import Trainer, seed_everything
#from pytorch_lightning.loggers.csv_logs import CSVLogger

# Plotting
import matplotlib.pyplot as plt

class LSTMTrainer(TrainerInterface):
    """
    Adapted from: https://github.com/adeboissiere/Handwriting-Prediction-and-Synthesis
    """

    def __init__(self):
        print("In ltsm con")

    def train(self, trainStrokeset, modelSaveLoc, hidden_size = 256, n_gaussians = 20, Kmixtures = 10, dropout = .2):
        torch.cuda.empty_cache()
        model = HandwritingSynthesisModel(hidden_size, n_gaussians, Kmixtures, dropout)

        if os.path.exists(modelSaveLoc):  # Load model if previously saved
            log.info(f"Loading model: {modelSaveLoc}")
            model.load_state_dict(torch.load(modelSaveLoc))
        else:
            model.eval()
            model = train_network(model, trainStrokeset, modelSaveLoc, epochs = 2, generate = True)

        self.trained_model = model

    def draw_strokes(data, factor=10, svg_filename='sample.svg'):
        min_x, max_x, min_y, max_y = get_bounds(data, factor)
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

    def line_plot(strokes, title):
        plt.figure(figsize=(20,2))
        eos_preds = np.where(strokes[:,-1] == 1)
        eos_preds = [0] + list(eos_preds[0]) + [-1] #add start and end indices
        for i in range(len(eos_preds)-1):
            start = eos_preds[i]+1
            stop = eos_preds[i+1]
            plt.plot(strokes[start:stop,0], strokes[start:stop,1],'b-', linewidth=2.0)
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def one_hot(s):
        #index position 0 means "unknown"
        alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        seq = [alphabet.find(char) + 1 for char in s]

        one_hot = np.zeros((len(s),len(alphabet)+1))
        one_hot[np.arange(len(s)),seq] = 1
        return one_hot

    def plot_heatmaps(Phis, Ws):
        plt.figure(figsize=(16,4))
        plt.subplot(121)
        plt.title('Phis', fontsize=20)
        plt.xlabel("time steps", fontsize=15)
        plt.ylabel("ascii #", fontsize=15)
    
        plt.imshow(Phis, interpolation='nearest', aspect='auto', cmap=cm.jet)
        plt.subplot(122)
        plt.title('Soft attention window', fontsize=20)
        plt.xlabel("time steps", fontsize=15)
        plt.ylabel("one-hot vector", fontsize=15)
        plt.imshow(Ws, interpolation='nearest', aspect='auto', cmap=cm.jet)

        display(plt.gcf())

    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    def gaussianMixture(y, pis, mu1s, mu2s, sigma1s, sigma2s, rhos):
        n_mixtures = pis.size(2)
    
        # Takes x1 and repeats it over the number of gaussian mixtures
        x1 = y[:,:, 0].repeat(n_mixtures, 1, 1).permute(1, 2, 0) 
        # print("x1 shape ", x1.shape) # -> torch.Size([sequence_length, batch, n_gaussians])
    
        # first term of Z (eq 25)
        x1norm = ((x1 - mu1s) ** 2) / (sigma1s ** 2 )
        # print("x1norm shape ", x1.shape) # -> torch.Size([sequence_length, batch, n_gaussians])
    
        x2 = y[:,:, 1].repeat(n_mixtures, 1, 1).permute(1, 2, 0)  
        # print("x2 shape ", x2.shape) # -> torch.Size([sequence_length, batch, n_gaussians])
    
        # second term of Z (eq 25)
        x2norm = ((x2 - mu2s) ** 2) / (sigma2s ** 2 )
        # print("x2norm shape ", x2.shape) # -> torch.Size([sequence_length, batch, n_gaussians])
    
        # third term of Z (eq 25)
        coxnorm = 2 * rhos * (x1 - mu1s) * (x2 - mu2s) / (sigma1s * sigma2s) 
    
        # Computing Z (eq 25)
        Z = x1norm + x2norm - coxnorm
    
        # Gaussian bivariate (eq 24)
        N = torch.exp(-Z / (2 * (1 - rhos ** 2))) / (2 * np.pi * sigma1s * sigma2s * (1 - rhos ** 2) ** 0.5) 
        # print("N shape ", N.shape) # -> torch.Size([sequence_length, batch, n_gaussians]) 
    
        # Pr is the result of eq 23 without the eos part
        Pr = pis * N 
        # print("Pr shape ", Pr.shape) # -> torch.Size([sequence_length, batch, n_gaussians])   
        Pr = torch.sum(Pr, dim=2) 
        # print("Pr shape ", Pr.shape) # -> torch.Size([sequence_length, batch])   
    
        if use_cuda:
            Pr = Pr.cuda()
    
        return Pr

    def loss_fn(Pr, y, es):
        loss1 = - torch.log(Pr + eps) # -> torch.Size([sequence_length, batch])    
        bernouilli = torch.zeros_like(es) # -> torch.Size([sequence_length, batch])
    
        bernouilli = y[:, :, 2] * es + (1 - y[:, :, 2]) * (1 - es)
    
        loss2 = - torch.log(bernouilli + eps)
        loss = loss1 + loss2 
        # print("loss shape", loss.shape) # -> torch.Size([sequence_length, batch])  
        loss = torch.sum(loss, 0) 
        # print("loss shape", loss.shape) # -> torch.Size([batch]) 
    
        return torch.mean(loss);

    def train_network(model, trainStrokeset, modelSaveLoc, epochs = 5, generate = True):
        data_loader = DataLoader(trainStrokeset, n_batch, sequence_length, 20, U_items=U_items) # 20 = datascale
    
        optimizer = optim.Adam(model.parameters(), lr=0.005)
    
        # A sequence the model is going to try to write as it learns
        c0 = np.float32(one_hot("writing is hard!"))
        c0 = torch.from_numpy(c0) 
        c0 = torch.unsqueeze(c0, 0) # torch.Size(n_batch, U_items, len(alphabet))
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
            data_loader.reset_batch_pointer()
        
            # Loop over batches
            for batch in range(data_loader.num_batches):
                # Loading a batch (x : stroke sequences, y : same as x but shifted 1 timestep, c : one-hot encoded character sequence ofx)
                x, y, s, c = data_loader.next_batch()
                x = np.float32(np.array(x)) # -> (n_batch, sequence_length, 3)
                y = np.float32(np.array(y)) # -> (n_batch, sequence_length, 3)
                c = np.float32(np.array(c))

                x = torch.from_numpy(x).permute(1, 0, 2) # torch.Size([sequence_length, n_batch, 3])
                y = torch.from_numpy(y).permute(1, 0, 2) # torch.Size([sequence_length, n_batch, 3])
                c = torch.from_numpy(c) # torch.Size(n_batch, U_items, len(alphabet))
            
                if use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                    c = c.cuda()
            
                # Forward pass
                es, pis, mu1s, mu2s, sigma1s, sigma2s, rhos = model.forward(x, c)
            
                # Calculate probability density and loss
                Pr = gaussianMixture(y, pis, mu1s, mu2s, sigma1s, sigma2s, rhos)
                loss = loss_fn(Pr,y, es)
            
                # Back propagation
                optimizer.zero_grad()
                loss.backward()
            
                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_threshold)
                optimizer.step()
            
                # Useful infos over training
                if batch % 10 == 0:
                    print("Epoch : ", epoch, " - step ", batch, "/", data_loader.num_batches, " - loss ", loss.item(), " in ", time.time() - start)
                    start = time.time()
                
                    # Plot heatmaps every 100 batch
                    if batch % 100 == 0:
                        print(s[0])
                        plot_heatmaps(model.Phis.transpose(0, 1).detach().numpy(), model.Ws.transpose(0, 1).detach().numpy())
                    
                    # Generate a sequence every 500 batch        
                    if generate and batch % 500 == 0 :
                        x0 = torch.Tensor([0,0,1]).view(1,1,3)

                        if use_cuda:
                            x0 = x0.cuda()
                    
                        for i in range(5):
                            sequence = model.generate_sequence(x0, c0, bias = 10)
                            print(sequence.shape)
                            draw_strokes_random_color(sequence, factor=0.5)
                    
                # Save loss per batch
                time_batch.append(epoch + batch / data_loader.num_batches)
                loss_batch.append(loss.item())
        
            # Save loss per epoch
            time_epoch.append(epoch + 1)
            loss_epoch.append(sum(loss_batch[epoch * data_loader.num_batches : (epoch + 1)*data_loader.num_batches-1]) / data_loader.num_batches)
        
            # Save model after each epoch
            log.info(f"Saving model after epoch {epoch}: {modelSaveLoc}")
            torch.save(model.state_dict(), modelSaveLoc)
        
        # Plot loss 
        plt.plot(time_batch, loss_batch)
        plt.plot(time_epoch, [loss_batch[0]] + loss_epoch, color="orange", linewidth=5)
        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.show()
        

        return model