import numpy as np
import sys

class Stump:
    def __init__(self, feat, p, theta, alpha):
        self.feat = feat
        self.p = p
        self.theta = theta
        self.alpha = alpha

    def evaluate(self, int_img_rep):
        eval_feature = self.feat.eval_learner(int_img_rep, self.p, self.theta)
        return self.alpha * eval_feature

class StrongLearner:

    def __init__(self):
        self.stumps = []
        self.big_theta = 0

    def add_stump(self, stump):
        self.stumps.append(stump)
    
    def remove_last_stump(self):
        self.stumps = self.stumps[:len(self.stumps) - 1]

    def evaluate(self, int_img_rep):
        evald = sum([stump.evaluate(int_img_rep) for stump in self.stumps])\
            + np.finfo(np.float64).eps + self.big_theta
        return np.sign(evald)

    def adjust_big_theta(self, int_img_rep, labels):
        evals = sum([stump.evaluate(int_img_rep) for stump in self.stumps])\
            + np.finfo(np.float64).eps
        preds = np.sign(evals)
        false_negs = np.extract(np.logical_and(labels == 1, labels != preds), evals)
        if len(false_negs) == 0:
            self.big_theta = 0
        else:
            self.big_theta = -0.999 * np.min(false_negs)

    def get_false_positive_rate(self, int_img_rep, labels):
        preds = self.evaluate(int_img_rep)
        fps = np.sum(np.logical_and(labels == -1, labels != preds))
        return fps / np.sum(labels == -1)

    def get_total_error_rate(self, int_img_rep, labels):
        preds = self.evaluate(int_img_rep)
        fps = np.sum(labels != preds)
        return fps / len(labels)

    def get_false_negative_rate(self, int_img_rep, labels):
        preds = self.evaluate(int_img_rep)
        fps = np.sum(np.logical_and(labels == 1, labels != preds))
        return fps / np.sum(labels == 1)
