import sys, os, click, glob

import pywritesmooth.Smooth.Smoother as sm
import pywritesmooth.TrainSmoother.LTSMTrainer as ltsm
import pywritesmooth.TrainSmoother.GANTrainer as gan
import pywritesmooth.Data.Stroke as stroke
import pywritesmooth.Data.StrokeSet as strokeset
import pywritesmooth.Data.StrokeDataset as sds

@click.command()
@click.option('-s', '--smooth', type=click.File('rb'), help = 'Image file of printed digits or letters in upper or lower case to be smoothed')
@click.option('-sm', '--smooth-model', default = 'gan', type=click.Choice(['gan', 'ltsm'], case_sensitive=False), help = 'Preferred smoothing model, options are GAN or LTSM')
@click.option('-t', '--train', type=click.STRING, help = 'Image file of printed digits or letters in upper or lower case to train the model(s)')
@click.option('-tm', '--train-models', multiple=True, type=click.Choice(['gan', 'ltsm'], case_sensitive=False), help = 'Models to be trained, options are GAN or LTSM')
def main(smooth = None, smooth_model = None, train = None, train_models = None):
    """The main routine.
    
    
    pywritesmooth --smooth <samplehw>  # Show to screen with default model
    pywritesmooth --smooth <samplehw> --smooth-model <model>  # Show to screen with specified model (GAN, LTSM, etc.)
    pywritesmooth --train <traindata> --train-models <model> <model> <etc>  # Train with specified models
    """

    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1

    try:
        if train is None and smooth is None:
            try:
                calledName = __loader__.fullname  # When called as a module
            except:
                calledName = os.path.basename(__file__)  # When called as a script

            print(__loader__, __name__, __package__, __spec__, __spec.parent, __file__)
            print("Usage: ", calledName, " --smooth <handwriting sample> --smooth-model <gan | ltsm>  --OR--")
            print("Usage: ", calledName, " --train <handwriting sample> --train-models <gan | ltsm>")

        if not train is None:
            if train_models is None:
                print("Please specify --train-models <model> switch when using --train")
                return EXIT_FAILURE
            else:
                hwData = sds.StrokeDataset()
                hwInput = glob.glob(train)

                writingSample = hwData.load(hwInput)
                models = []

                for modelName in train_models:
                    if modelName == 'ltsm':
                        models.append(ltsm.LTSMTrainer())
                    if modelName == 'gan':
                        models.append(gan.GANTrainer())

                #models = BuildModels(hw, models)
                #TestModels(models)

        if not smooth is None:
            if smooth_model is None:
                print("Please specify --smooth_model <gan | ltsm> switch when using --smooth")
                return EXIT_FAILURE
            else:
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
    sys.exit(main())