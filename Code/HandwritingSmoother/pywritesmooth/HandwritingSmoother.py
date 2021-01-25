# Basics
import sys, os, click, glob, logging as log

# Project
import pywritesmooth.Smooth.Smoother as sm
import pywritesmooth.TrainSmoother.LSTMTrainer as lstm
import pywritesmooth.TrainSmoother.GANTrainer as gan
import pywritesmooth.Data.Stroke as stroke
import pywritesmooth.Data.StrokeSet as strokeset
import pywritesmooth.Data.StrokeDataset as sds

@click.command()
@click.option('-s', '--smooth', type=click.File('rb'), help = 'Image file of printed digits or letters in upper or lower case to be smoothed')
@click.option('-sm', '--smooth-model', default = 'lstm', type=click.Choice(['gan', 'lstm'], case_sensitive=False), help = 'Preferred smoothing model, options are GAN or LSTM')
@click.option('-ss', '--smooth-sample', type=click.STRING, help="Filename of a writing sample in online (XML) format similar to the structure of the IAM dataset")
@click.option('-t', '--train', type=click.STRING, help = 'Image file(s) of printed digits or letters in upper or lower case to train the model(s)')
@click.option('-tm', '--train-models', multiple=True, type=click.Choice(['gan', 'lstm'], case_sensitive=False), help = 'Models to be trained, options are GAN or LSTM')
@click.option('-m', '--saved-model', type=click.STRING, help = 'Filename of a HandwritingSynthesisModel for saving/loading')
@click.option('-p', '--pickled-data', type=click.STRING, help = 'Filename of a StrokeDataset for saving/loading in Python pickle format')
@click.option('-l', '--log-file', type=click.STRING, help = 'Filename of the log file')
@click.option('-ll', '--log-level', type=click.STRING, help = 'Logging level: critical, error, warning, info, or debug')
@click.option('-is', '--image-save', type=click.STRING, help = 'Location to save plot images; file name given will have numbers appended, i.e. images\phi will become images\phi1.png, phi2.png, etc.')
@click.option('-id', '--image-display', is_flag=True, help = 'To display or not to display.. the plots')
@click.option('-hws', '--hw-save', type=click.STRING, help = 'Location to save handwriting images; file name given will have numbers appended, i.e. images\hw will become images\hw1.png, hw2.png, etc.')
@click.option('-hs', '--handwriting-save', is_flag=True, help = 'To save or not to save.. the samples (of which there could be a ton)')
@click.option('-gs', '--generated-save', is_flag=True, help = 'To save or not to save.. the generated strokes (of which there could be a ton)')
@click.option('-w', '--write', type=click.STRING, help="Generate handwriting from a text string: max of 80 chars")
@click.option('-tm', '--test-model', is_flag=True, help = 'Flag if you want to automatically run the handwriting generatation tests, which will save as svg files')
def main(smooth = None, smooth_model = None, train = None, train_models = None, saved_model = None, pickled_data = None, 
         log_file = None, log_level = None, image_save = None, image_display = False, hw_save = None, 
         handwriting_save  = False, generated_save = False, write = None, test_model = False, smooth_sample = None):
    """The main routine.
    
    
        Some execution examples:
        pywritesmooth --smooth <samplehw>  # Show to screen with default model
        pywritesmooth --smooth <samplehw> --smooth-model <model>  # Show to screen with specified model (GAN, LTSM, etc.)
        pywritesmooth --saved-model <filename> --train <traindata> --train-models <model> <model> <etc>  # Train with specified models
        pywritesmooth --pickled-data <filename> --smooth <samplehw> --smooth-model <model>  # Show to screen with specified model (GAN, LTSM, etc.)
        pywritesmooth --pickled-data <filename> --saved-model <filename> --train <traindata> --train-models <model> <model> <etc>  # Train with specified models

        This application provides functionality related to handwriting.  Specifically, once it has been trained with a number of handwriting samples 
        taken from the IAM online database, the model will be able to generate handwritten text on demand from ascii text.  In addition,
        if a handwriting sample is supplied in the same IAM format, it will use the trained model to smooth the handwriting while still retaining
        its unique stylistic characteristics.

        At this time, only the LSTM model is implemented; however, the program is written to accomodate any number of smoothing models if they are
        written in the future.
    """
    
    # Constants
    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1

    try:
        # Default log file
        hw_log_file = os.path.join(".", "pywritesmooth.log")
        if not log_file is None:
            hw_log_file = log_file
        os.makedirs(os.path.dirname(hw_log_file), exist_ok=True)

        # Logging level
        logging_level = log.INFO
        if not log_level is None:
            ll = log_level.lower()
            logging_level = log.CRITICAL if ll == 'critical' else \
                            log.WARNING if ll == 'warning' else \
                            log.ERROR if ll == 'error' else \
                            log.DEBUG if ll == 'debug' else \
                            log.INFO

        log.basicConfig(filename=hw_log_file, level=logging_level, 
                        format=r'%(asctime)s %(levelname)s (%(filename)s/%(funcName)s:%(lineno)d): %(message)s', 
                        datefmt='%d-%b-%y %H:%M:%S')
        log.debug("Starting app")

        if train is None and smooth is None:
            try:
                calledName = __loader__.fullname  # When called as a module
                log.debug("Running as a module")
            except:
                calledName = os.path.basename(__file__)  # When called as a script
                log.debug("Running as a script")

            #print(__loader__, __name__, __package__, __spec__, __spec.parent, __file__)
            print("Usage: ", calledName, " --smooth <handwriting sample> --smooth-model <gan | lstm>  --OR--")
            print("Usage: ", calledName, " --train <handwriting sample> --train-models <gan | lstm>")

        # Default model file
        hwModelSave = os.path.join(".", "hwSynthesis.model")
        if not saved_model is None:
            hwModelSave = saved_model

        # Default pickled data file
        hwDataSave = os.path.join(".", "hwData.pkl")
        if not pickled_data is None:
            hwDataSave = pickled_data

        # Image save location
        hwPlotImages = os.path.join(".", "plots", "phi")
        if not image_save is None:
            hwPlotImages = image_save

        # Sample save location
        hwSaveSamples= os.path.join(".", "samples", "hw")
        if not hw_save is None:
            hwSaveSamples = hw_save

        # Train models of interest
        if not train is None:
            if train_models is None:
                print("Please specify --train-models <model> switch when using --train")
                log.critical("Training switch missing or incorrect; exiting")
                return EXIT_FAILURE
            else:
                log.info(f"Training model data: {train}")
                log.info(f"Training model args: {train_models}")
                hwInput = glob.glob(train)
                log.debug(f"Input files: {hwInput}")

                writingSample = sds.StrokeDataset(hwInput, hwDataSave)
                models = []

                for modelName in train_models:
                    if modelName == 'lstm':
                        models.append(lstm.LSTMTrainer(hwModelSave, image_display, 
                                                       hwPlotImages, handwriting_save, 
                                                       hwSaveSamples, generated_save))
                    if modelName == 'gan':
                        models.append(gan.GANTrainer())

                models = BuildModels(writingSample, models)

                if test_model:
                    WriteText(models)

        # Smooth handwriting sample provided by the user
        if not smooth is None:
            if smooth_model is None:
                print(r"Please specify --smooth_model <gan | ltsm> switch when using --smooth")
                log.critical(r"Smoothing switch missing or incorrect; exiting")
                return EXIT_FAILURE
            elif smooth_sample is None:
                print(r"Please specify --smooth-sample <sample filename>")
                log.critical(r"Smoothing sample missing or incorrect; exiting")
                return EXIT_FAILURE
            else:
                log.info(f"Smoothing models selected: {smooth_model}")
                SmoothWriting(smooth_sample, models)

        # Generate handwritten text supplied by user
        if not write is None:
            WriteText(models, write)

    except NotImplementedError as nie:
        print("Ran into some code that needs implementation: ", nie)
        log.critical(f"Ran into some code that needs implementation; exiting", exc_info=True)
        return EXIT_FAILURE
    except:
        print("Exception: ",sys.exc_info())
        log.critical(f"Unexpected exception", exc_info=True)
        return EXIT_FAILURE

    log.info("Exiting normally")
    return EXIT_SUCCESS

def BuildModels(hw, modelsToTrain):
    """BuildModels

       Have each model train itself or load a trained model.
    """

    for model in modelsToTrain:
        model.train(hw)

    return modelsToTrain

def WriteText(trainedModels, genList = ['Sample text']):
    """WriteText

       Test the models available by having them generate handwriting from sample text.
    """

    for model in trainedModels:
        for text in genList:
            model.asHandwriting(text)

def SmoothWriting(hwSample, models):
    """SmoothWriting

       Smooth a handwriting sample.  The sample must be in the IAM online data format (XML)
       at this time.  The result will be saved to an SVG file using the path specified in the
       *generated-save* flag.

    """
    for model in models:
        model.smoothHandwriting(hwSample)   

if __name__ == "__main__":
    sys.exit(main())