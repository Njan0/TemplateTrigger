import threading
import numpy as np
import cv2
from mss import mss

MAX_DIFF = 3*255*255


class Observer:
    """
    Observer which calls registered functions when a template image is found on screen.
    """
    ON_FOUND = 1
    ON_LOST = 2
    ON_MOVE = 3
    TRIGGERS = [ON_FOUND, ON_LOST, ON_MOVE]

    def __init__(self, template, threshold=0., bbox=None):
        """
        :param template: Template image to find on screen
        :param threshold: Minimum similarity of match to template
        :param bbox: Search area of screen
        """
        dim = template.shape[2]
        if dim == 4:
            color, a = np.split(template, [-1], 2)
            self._template_image = color
            self._template_mask = np.float32(np.clip(a * (1. / 255), 0, 1))
            self._sq_alpha = np.sum(np.square(self._template_mask))
        elif dim == 3:
            self._template_image = template
            self._template_mask = None
            self._sq_alpha = np.prod(self._template_image.shape[:2])
        else:
            raise Exception("Invalid dimension of template.")

        self._threshold = threshold
        self._bbox = bbox
        self._lastFound = None
        self._fun_register = {trigger: [] for trigger in Observer.TRIGGERS}
        self._current_thread = None

    def check(self):
        """
        Search screen for template image.

        :return: Position of best match and corresponding similarity
        """
        if self._sq_alpha == 0:
            return (0, 0), 1

        # get image of current screen
        with mss() as sct:
            if self._bbox is None:
                screen = sct.grab(sct.monitors[1])
            else:
                screen = sct.grab(self._bbox)
            img = np.array(screen)[..., :3]

        # perform template match
        result = cv2.matchTemplate(img, self._template_image, cv2.TM_SQDIFF, None, self._template_mask)
        val, _, location, _ = cv2.minMaxLoc(result)

        # calculate similarity
        similarity = np.clip(1 - val / MAX_DIFF / self._sq_alpha, 0, 1)

        return location, similarity

    def update(self):
        """
        Search screen for template image and call
        registered functions according to the results.
        """
        location, sim = self.check()
        if sim >= self._threshold:
            # image found
            if self._lastFound is None:
                # image not found in last step
                self._trigger(Observer.ON_FOUND, location)
            elif location != self._lastFound:
                # image moved
                self._trigger(Observer.ON_MOVE, location)

            self._lastFound = location
        elif self._lastFound is not None:
            # image lost
            self._trigger(Observer.ON_LOST, self._lastFound)
            self._lastFound = None

    def register(self, fun, trigger):
        """
        Register a function.

        :param fun: Function to be called if a event is triggered
        :param trigger: The event to be observed
        """
        if trigger in Observer.TRIGGERS:
            self._fun_register[trigger].append(fun)
        else:
            raise Exception("Unknown trigger: {}".format(trigger))

    def unregister(self, fun, trigger):
        """
        Unregister a function.

        :param fun: Function to be removed
        :param trigger: The event which would call the function
        """
        if trigger in Observer.TRIGGERS:
            self._fun_register[trigger].remove(fun)
        else:
            raise Exception("Unknown trigger: {}".format(trigger))

    def _trigger(self, trigger, *args):
        # trigger all function registered to corresponding event
        for fun in self._fun_register[trigger]:
            fun(*args)

    def run(self, update_period):
        """
        Run the observer in another thread to
        repeatedly check for events.

        :param update_period: Time between checks
        """
        if self._current_thread is not None:
            raise Exception("Observer is already running.")

        stop_event = threading.Event()

        def async_run():
            while not stop_event.wait(update_period):
                self.update()

        thread = threading.Thread(target=async_run)
        self._current_thread = thread, stop_event
        thread.start()

    def stop(self):
        """
        If the observer runs in another thread stop it.
        """
        if self._current_thread is None:
            raise Exception("Observer is not running.")

        thread, stop_event = self._current_thread
        stop_event.set()
        thread.join()
        self._current_thread = None
