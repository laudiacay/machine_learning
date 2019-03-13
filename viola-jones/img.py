from PIL import Image
import numpy as np

class Img:
   
    def __init__(self):
        pass
         
    def load_data(self, faces_dir, background_dir, N=2000):
        '''
        faces_dir: string, file location of the faces directory
        background_dir: string, file location of the background directory
        N: number of faces and backgrounds to load
        Returns: a tuple of numpy matrices
        - a matrix of size N x 64 x 64 of the images
        - a matrix of size N x 1 of the class labels(+1 for a face, -1 for background)
        '''

        images = np.empty((2*N, 64, 64), dtype=np.float64)
        labels = np.empty(2*N, dtype=np.int8)
        
        shuffle_order = np.arange(len(images))
        np.random.shuffle(shuffle_order)
        
        for i in range(N):
            with Image.open('{}/face{}.jpg'.format(faces_dir, str(i))) as imgfile:
                images[shuffle_order[i]] = imgfile.convert('L')
            
            labels[shuffle_order[i]] = 1
            
            with Image.open('{}/{}.jpg'.format(background_dir, str(i))) as imgfile:
                images[shuffle_order[N+i]] = imgfile.convert('L')
            
            labels[shuffle_order[N+i]] = -1

        return images, labels

    def load_trial_image(self, filename):
        return Image.open(filename).convert('L')

    def compute_integral_image(self, img):
        s = np.empty(img.shape)
        ii = np.empty(img.shape)

        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                s[x][y] = s[x][y-1] + img[x][y] if y > 0 else img[x][y]
                ii[x][y] = ii[x-1][y] + s[x][y] if x > 0 else s[x][y]
        return ii


    def compute_integral_images(self, imgs):
        '''
        imgs: numpy matrix of size N x 64 x 64, where N = total number of images
        Returns: a matrix of size N x 64 x 64 of the integral image representation
        '''
        iis = np.empty(imgs.shape)
        for i in range(len(imgs)):
            iis[i] = self.compute_integral_image(imgs[i])
        return iis

    def intensity(self, img, feature, dark):
        if dark:
            x1, y1 = feature.d1
            x2, y2 = feature.d2
        else:
            x1, y1 = feature.l1
            x2, y2 = feature.l2
        return (img[x2][y2] + img[x1][y1] - img[x2][y1] - img[x1][y2])

    def mark_image(self, img, x, y):
        for i in range(64):
            img[x+i, y, 0] = 255
            img[x+i, y, 1] = 0
            img[x+i, y, 2] = 0
            img[x+i, y+1, 0] = 255
            img[x+i, y+1, 1] = 0
            img[x+i, y+1, 2] = 0
            img[x+i, y+64, 0] = 255
            img[x+i, y+64, 1] = 0
            img[x+i, y+64, 2] = 0
            img[x+i, y+64+1, 0] = 255
            img[x+i, y+64+1, 1] = 0
            img[x+i, y+64+1, 2] = 0
            img[x, y+i, 0] = 255
            img[x, y+i, 1] = 0
            img[x, y+i, 2] = 0
            img[x+64, y+i, 0] = 255
            img[x+64, y+i, 1] = 0
            img[x+64, y+i, 2] = 0
            img[x+1, y+i, 0] = 255
            img[x+1, y+i, 1] = 0
            img[x+1, y+i, 2] = 0
            img[x+64+1, y+i, 0] = 255
            img[x+64+1, y+i, 1] = 0
            img[x+64+1, y+i, 2] = 0
