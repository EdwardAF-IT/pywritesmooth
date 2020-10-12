import sys

from pywritesmooth.Smooth import *
from pywritesmooth.TrainSmoother import *
from pywritesmooth.Utility import *

def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    print("This is the main routine.")
    print("It should do something interesting.")
    print(args)

    # Do argument parsing here (eg. with argparse) and anything else
    # you want your project to do. Return values are exit codes.
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))