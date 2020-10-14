import sys
import click

from pywritesmooth.TrainSmoother import *
from pywritesmooth.Utility import *
import pywritesmooth.Smooth.Smoother as sm
#import pywritesmooth.TrainSmoother.GANTrainer as gan
#import pywritesmooth.TrainSmoother.StyleTransferTrainer as st
#import pywritesmooth.Utility.HandwritingData as hw
#import pywritesmooth.Utility.Log as log


@click.command()
@click.option('-s', '--smooth', type=click.File('rb'), help = 'Image file of printed digits or letters in upper or lower case to be smoothed')
@click.option('-sm', '--smooth-model', default = 'gan', type=click.Choice(['gan', 'st'], case_sensitive=False), help = 'Preferred smoothing model, options are GAN or ST (StyleTransfer)')
@click.option('-t', '--train', type=click.File('rb'), help = 'Image file of printed digits or letters in upper or lower case to train the model(s)')
@click.option('-tm', '--train-models', multiple=True, type=click.Choice(['gan', 'st'], case_sensitive=False), help = 'Models to be trained, options are GAN or ST (StyleTransfer)')
def main(smooth = None, smooth_model = None, train = None, train_models = None):
    """The main routine."""

    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1

    try:
        switcher = {
            'gan': GANTrainer.GANTrainer(),
            #'st': StyleTransferTrainer()
            }
        return 0
        if not train is None:
            if train_models is None:
                print("Please specify --train-models <model> switch when using --train")
                return EXIT_FAILURE
            else:
                trainData = hw.HandwritingData(train)
                models = []

                for modelName in train_models:
                    models.append(switcher.get(modelName))

                models = BuildModels(hw, models)
                TestModels(models)

        if not smooth is None:
            hw = Handwriting(smooth_model)
            SmoothWriting(hw, switcher.get(smooth_model))
    except NotImplementedError as nie:
        print("Ran into some code that needs implementation: ", nie)
        return EXIT_FAILURE
    except:
        print("Exception: ", sys.exc_info())
        return EXIT_FAILURE

    return EXIT_SUCCESS

def BuildModels(hw, modelsToTrain):
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
    sm = Smoother(hwSample, modelToUse)

    sm.merge()
    display(sm.merged())
    sm.save()

if __name__ == "__main__":
    sys.exit(main())