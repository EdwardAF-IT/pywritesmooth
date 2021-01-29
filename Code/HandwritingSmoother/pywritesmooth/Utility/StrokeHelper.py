import sys, numpy as np, logging as log

class StrokeHelper(object):
    """StrokeHelper

       Helper methods for Stroke data types.
    """

    def init(self):
        log.debug("Init")
        self.name = "StrokeHelper"

    def __init__(self):
        log.debug("Default contructor")
        self.init()

    def normalize_points(self, points):
        """normalizePoints

           Return the collection of stroke points as a 2D numpy array where the minimum point valules are 0.
        """

        norm = np.array(points)
        normX = norm[:,0].astype(float)
        normY = norm[:,1].astype(float)
        normT = norm[:,2].astype(float)

        new = np.empty([len(normX), 3])
        new[:,0] = normX - min(normX)
        new[:,1] = normY - min(normY)
        new[:,2] = normT - min(normT)

        log.debug(f"Normalized array: {new}")
        return(new)

    def progress(count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        percents = 100.0 if bar_len - filled_len == 0 else percents       # Show 100% if complete
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()