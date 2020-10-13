from sklearn.model_selection import train_test_split

class Handwriting(object):
    """Handwriting

       This class manages input data, which for this application will take the form of an image of printed uppercase, lowercase, and numeric characters.  The image format should be a common format such as jpg, png, or gif.

       Additionally, the classes handles tasks like splitting images into individual characters and normalizing their sizes.
    """

    def __init__(self, inputFileName):
        self.rawImage = load(inputFileName)
        self.letters = splitletters(self.rawImage)
        self.letters = normalizeSizes(self.letters)
        trainTestSplit()
        self.trainVector = vectorize(self.trainLetters)
        self.testVector = vectorize(self.testLetters)

    def load(inputFileName):
        # Load image from filename

        raise NotImplementedError

    def splitLetters(rawImage):
        letterList = []

        # Split into letters

        return letterList
        raise NotImplementedError

    def normalizeSizes(letters):
        normalizedLetters = []

        for letter in letters:
            normalizedLetters.append(normalizeLetter(letter))

        return normalizedLetters

    def normalizeLetter(letter, size = (28, 28)):
        # Normalize character size to 28x28 pixels

        return NotImplementedError

    def stitchTogether(imageType = "png"):
        # Put the list of indvidual letters together into an image

        return NotImplementedError

    def vectorize():
        # Convert the list of letters into a vectorized matrix of features

        return NotImplementedError

    def trainTestSplit(splitRatio = .7):
        self.trainLetters, self.testLetters = train_test_split(self.letters, train_size=splitRatio, random_state=19)