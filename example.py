import time

import cv2
from tt import Observer

if __name__ == '__main__':
    template = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)

    obs = Observer(template, 0.95)
    obs.register(lambda location: print("Found at {}".format(location)), Observer.ON_FOUND)
    obs.register(lambda location: print("Moved to {}".format(location)), Observer.ON_MOVE)
    obs.register(lambda location: print("Lost at {}".format(location)), Observer.ON_LOST)

    # async run
    obs.run(1)
    time.sleep(5)
    obs.stop()
