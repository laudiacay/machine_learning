import numpy as np
from img import Img

iimg = Img()

class Cascade:
    def __init__(self):
        self.strong_learners = []

    def add_strong_learner(self, strong_learner):
        self.strong_learners.append(strong_learner)

    def evaluate_images(self, int_imgs):
        predictions = np.full(int_imgs.shape[0], 1)
        for sl in self.strong_learners:
            for i, val in enumerate(sl.evaluate(int_imgs)):
                if val == -1:
                    predictions[i] = -1
        return predictions

    def evaluate_image(self, img):
        int_imgs = np.array([iimg.compute_integral_image(img)])
        return self.evaluate_images(int_imgs)[0]

    def get_correct_bg_indices(self, int_imgs, labels):
        predictions = self.evaluate_images(int_imgs)
        is_correct_and_bg = predictions == labels
        return np.argwhere(is_correct_and_bg)

    def calc_success_rate(self, int_imgs, labels):
        is_right = np.equal(self.evaluate_images(int_imgs), labels)
        return sum([1 for i in is_right if i]) / len(int_imgs)
