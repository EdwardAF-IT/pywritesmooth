import sys
import click

from pywritesmooth.Smooth import *
from pywritesmooth.TrainSmoother import *
from pywritesmooth.Utility import *

@click.command()
@click.option('-s', '--smooth', type=click.File('rb'), help = 'Image file of printed digits or letters in upper or lower case to be smoothed')
@click.option('-sm', '--smooth-model', default = ['GAN'], type=click.Choice(['GAN', 'ST'], case_sensitive=False), help = 'Preferred smoothing model, options are GAN or ST (StyleTransfer)')
@click.option('-t', '--train', type=click.File('rb'), help = 'Image file of printed digits or letters in upper or lower case to train the model(s)')
@click.option('-tm', '--train-models', default = ['GAN'], type=click.Choice(['GAN', 'ST'], case_sensitive=False), multiple=True, help = 'Models to be trained, options are GAN or ST (StyleTransfer)')
def main(smooth = None, smooth_model = None, train = None, train_models = None):
    """The main routine.
    
    
    pywritesmooth --smooth <samplehw>  # Show to screen with default model
pywritesmooth --smooth <samplehw> --smooth-model <model>  # Show to screen with specified model (GAN, Style-transfer, etc.)
pywritesmooth --train <traindata> --train-models <model> <model> <etc>  # Train with specified models
    """

    switcher = {
        'gan': GANTrainer,
        'st': StyleTransferTrainer
        }

    if not train is None:
        hw = Handwriting(train)
        models = []

        for modelName in train_models:
            models.append(switcher.get(modelName))

        models = BuildModels(hw, models)
        TestModels(models)

    if not smooth is None:
        hw = Handwriting(smooth_model)
        SmoothWriting(hw, switcher.get(smooth_model))

    return 0

def BuildModels(hw, modelsToTrain):
    """
    Foreach model type:
	    Input --> Clean and transform input  --> Cleaned input
	    Cleaned input --> Identify features --> Feature vector
	    Cleaned input, feature vector --> Train the model --> Trained model
	    Trained model --> Test the model --> Model metrics/error
    """

    for model in modelsToTrain:
        model.train(hw.trainVector)
        model.save()

    return modelsToTrain

def TestModels(trainedModels):
    for model in trainedModels:
        print(model.getError(model.test()))

def LoadModels(modelsToLoad):
    for model in modelsToLoad:
        model.load()

    return modelsToLoad

def SmoothWriting(hwSample, modelToUse):
    """
    Input Writing to Smooth --> Clean and transform input  --> Cleaned input
    Cleaned input, Trained model --> SmoothHandwriting --> Smoothed writing
    Smoothed writing --> OutputWriting --> Image file

    """

    sm = Smoother(hwSample, modelToUse)

    sm.merge()
    display(sm.merged())
    sm.save()

if __name__ == "__main__":
    main()