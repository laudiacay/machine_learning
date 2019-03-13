import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from img import Img
import copy, time
from multiprocessing import Pool
from functools import partial
from sklearn.tree import DecisionTreeClassifier

iimg = Img()

class Feature(object):
     
    feature_list = []
    cur_fl = []

    frame_size = 64
    start_width_height = 0
    stop_width_height = frame_size//2
    stride_width_height = 3
    stride_x_y = 3

    def __init__(self, x1_d, y1_d, x2_d, y2_d, x1_l, y1_l, x2_l, y2_l):
        self.d1 = x1_d, y1_d
        self.d2 = x2_d, y2_d
        self.l1 = x1_l, y1_l
        self.l2 = x2_l, y2_l
        self.clf = DecisionTreeClassifier(max_depth = 1, random_state = 1)
    
    # the viola jones paper uses only rectangles of the same shape that are side by side. so i'm just going to do that.
    @classmethod
    def gen_feature_list(cls):
        '''
        Returns: list of relevant pixel locations for each feature.
            The ith index of this list should be a list of the four relevant
            pixel locations needed to compute the ith feature
        '''
        for w in range(cls.start_width_height, cls.stop_width_height, cls.stride_width_height):
            for h in range(cls.start_width_height, cls.stop_width_height, cls.stride_width_height):
                for x in range(0, cls.frame_size - 2*w, cls.stride_x_y):
                    for y in range(0, cls.frame_size - 2*h, cls.stride_x_y):
                        # black on left
                        cls.feature_list.append(Feature(x, y, x+w-1, y+h-1, 
                                    x+w, y, x+2*w-1, y+h-1))
                        # black on top
                        cls.feature_list.append(Feature(x, y, x+w-1, y+h-1, 
                                    x, y+h, x+w-1, y+2*h-1))

    @classmethod
    def reset_cur_fl(cls):
        cls.cur_fl = copy.deepcopy(cls.feature_list)
        return

    def plot_rect(self):
        arr = np.full((64, 64), 0.5)
        
        x1_d, y1_d = self.d1
        x2_d, y2_d = self.d2
        x1_l, y1_l = self.l1
        x2_l, y2_l = self.l2

        for x in range(x1_d, x2_d + 1):
            for y in range(y1_d, y2_d + 1):
                arr[x][y] = 0.0
        for x in range(x1_l, x2_l + 1):
            for y in range(y1_l, y2_l + 1):
                arr[x][y] = 1.0
        
        plt.imshow(arr)
        plt.show()

    def find_area(self):
        return (self.d1[0] - self.d2[0]) * (self.d1[1] - self.d2[1])

    def compute_feature(self, int_img_rep):
        '''
        int_img_rep: the N x 64 x 64 numpy matrix of the integral image representation
        feat_lst: list of features
        feat_idx: integer, index of a feature (in feat_lst)
        Returns: an N x 1 numpy matrix of the feature evaluations for each image
        '''
        area = self.find_area()
        
        def feat_eval(img):
            return (iimg.intensity(img, self, True) - iimg.intensity(img, self, False))/area
        return np.array([feat_eval(img) for img in int_img_rep]).reshape(-1, 1)

    def opt_p_theta(self, int_img_rep, labels, weights):
        '''
        int_img_rep: the N x 64 x 64 numpy matrix of the integral image representation
        feat_lst: list of features
        weights: an N x 1 matrix containing the weights for each datapoint
        feat_idx: integer, index of the feature to compute on all of the images
        Returns: the optimal theta and p values for the given feat_idx
        '''
        
        feature_eval = self.compute_feature(int_img_rep)
        self.clf.fit(feature_eval, labels, sample_weight=weights)
        

    def eval_learner(self, int_img_rep, p, theta):
        feature_eval = self.compute_feature(int_img_rep)
        return self.clf.predict(feature_eval)
        
    def error_rate(self, int_img_rep, labels, weights, p, theta):
        predictions = self.eval_learner(int_img_rep, p, theta)
        
        incorrect = predictions != labels
        weighted_error = np.dot(weights, incorrect) / np.sum(weights)
        return weighted_error
    
    def error_rate_lambda(int_img_rep, labels, weights, x):
        return x.error_rate(int_img_rep, labels, weights, None, None)
     
    @classmethod
    def opt_weaklearner(cls, int_img_rep, labels, weights):
        with Pool(5) as p:
            start = time.time()
            [x.opt_p_theta(int_img_rep, labels, weights) for x in cls.cur_fl]
            end = time.time()
            print((end-start)/60)
            start = time.time()
            error_rates = np.array(p.map(partial(partial(partial(cls.error_rate_lambda, int_img_rep), labels), weights),
                tqdm(cls.cur_fl)))
            end = time.time()
            print((end-start)/60)
        
        best_ind = np.argmin(error_rates)
        best_t, best_p = None, None
        best_error = error_rates[best_ind]
        return best_ind, best_t, best_p, best_error
