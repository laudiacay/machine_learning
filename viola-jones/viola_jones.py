import numpy as np
import math, copy
from img import Img
from feature import Feature
from stronglearner import StrongLearner, Stump
from cascade import Cascade
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from multiprocessing import Pool

        
def adaboost_round(int_images, labels, round_num):
    
    Feature.reset_cur_fl()
    weights = np.full(len(int_images), 1/len(images), dtype=np.float64)
    final = StrongLearner()
    false_positive_rate = 1.0

    while false_positive_rate > 0.3 or len(final.stumps) < 3:
        print('continuing round {}'.format(str(round_num)))

        best_ind, t, p, err = Feature.opt_weaklearner(int_images, labels, weights)
        
        best_feature = Feature.cur_fl[best_ind]
        
        if err == 0:
            err += np.finfo(np.float64).eps
        if err == 1:
            err -= np.finfo(np.float64).eps

        a = 1/2 * math.log((1-err)/err)

        final.add_stump(Stump(best_feature, p, t, a))
        
        preds = best_feature.eval_learner(int_images, p, t)
        misses = preds != labels
        
        weights *= np.exp(misses * a)
        weights /= sum(weights)
        
        Feature.cur_fl = Feature.cur_fl[:best_ind] + Feature.cur_fl[best_ind + 1:]
        
        final.adjust_big_theta(int_images, labels)
        false_positive_rate = final.get_false_positive_rate(int_images, labels)
        false_negative_rate = final.get_false_negative_rate(int_images, labels)
        
        print('err:', final.get_total_error_rate(int_images, labels))
        print('false positive rate:', str(false_positive_rate))
        print('false negative rate:', str(false_negative_rate))
    return final

RESTORE_PARTIAL_CASCADE = False

def compute_cascade(train_images, train_labels, int_imgs, cascade=None):
    count = 1 if cascade == None else len(cascade.strong_learners) + 1
    if cascade == None:
        cascade = Cascade()
        success_rate_overall = 0.0
    else:
        success_rate_overall = cascade.calc_success_rate(train_images, train_labels)
    last_success_rate = 0.05
    print(success_rate_overall)
    while success_rate_overall < 0.995 and success_rate_overall != last_success_rate:
        
        print('***** starting round {} *****'.format(str(count)))
        
        cascade.add_strong_learner(adaboost_round(train_images, train_labels, count))
        
        correct_bg_indices = cascade.get_correct_bg_indices(train_images, train_labels)
        
        train_images = np.delete(train_images, correct_bg_indices, axis=0)
        train_labels = np.delete(train_labels, correct_bg_indices, axis=0)
        
        last_success_rate = success_rate_overall
        success_rate_overall = cascade.calc_success_rate(int_imgs, labels)
        print('overall success rate after this round: {}'.format(str(success_rate_overall)))
        print('images left: {}'.format(str(train_images.shape[0])))

        with open('partial_save_data.bin', 'wb') as f:
            pickle.dump(cascade, f)
        with open('partial_save_images.bin', 'wb') as f:
            pickle.dump(train_images, f)
        with open('partial_save_labels.bin', 'wb') as f:
            pickle.dump(train_labels, f)

        print('dumped partially trained model to file')
        
        count += 1

    if success_rate_overall >= 0.99:
        print('at least 99% of images classified correctly.')
    else:
        print('process failed to converge. success rate:', success_rate_overall)
    return cascade

RESTORE_CASCADE = False 

Feature.gen_feature_list()
img = Img()

if not RESTORE_CASCADE:
    images, labels = img.load_data('faces', 'background', N=2000)
    int_imgs = img.compute_integral_images(images)
    if RESTORE_PARTIAL_CASCADE:
        with open('partial_save_data.bin', 'rb') as f:
            cascade = pickle.load(f)
        with open('partial_save_images.bin', 'rb') as f:
            train_images = pickle.load(f)
        with open('partial_save_labels.bin', 'rb') as f:
            train_labels = pickle.load(f)
    else:
        train_images, train_labels = copy.deepcopy(int_imgs), copy.deepcopy(labels)
        cascade = None
    cascade = compute_cascade(train_images, train_labels, int_imgs, cascade=cascade)

    with open('save_data.bin', 'wb') as f:
        pickle.dump(cascade, f)
    print('dumped trained model to file')
else:
    with open('save_data.bin', 'rb') as f:
        cascade = pickle.load(f)


RESTORE_SQUARES = False
test_img = np.array(img.load_trial_image('test_img.jpg'))
marked_img = np.array(img.load_trial_image('test_img.jpg').convert('RGB'))

if not RESTORE_SQUARES:
    image_slices = []
    image_indices = []

    for x in tqdm(range(0, test_img.shape[0] - 64, 4)):
        for y in range(0, test_img.shape[1] - 64, 4):
            image_slices.append(test_img[x:x+64, y:y+64])
            image_indices.append((x,y))

    with Pool(5) as p:
        values = p.map(cascade.evaluate_image, tqdm(image_slices))
    where_to_put_squares = []

    for i in tqdm(range(len(values))):
        if values[i] == 1:
            where_to_put_squares.append(image_indices[i])

    with open('squares.bin', 'wb') as f:
        pickle.dump(where_to_put_squares, f)
else:
    with open('squares.bin', 'rb') as f:
        where_to_put_squares = pickle.load(f)

for loc in where_to_put_squares:
    img.mark_image(marked_img, loc[0], loc[1])

plt.axis('off')
plt.imshow(marked_img)
plt.savefig('test2.png', bbox_inches='tight')
