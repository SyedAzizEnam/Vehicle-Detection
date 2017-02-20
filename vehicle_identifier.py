import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip
import collections
import glob
import os
import collections

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()

heat_maps = collections.deque(maxlen=10)

def loadimages(basedir):

    image_types = os.listdir(basedir)
    if '.DS_Store' in image_types:
        image_types.remove('.DS_Store')
    image_paths = []
    for imtype in image_types:
        image_paths.extend(glob.glob(basedir+'/'+imtype+'/*'))

    return image_paths

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32,32)):
    channel1 = cv2.resize(img[:,:,0], size).ravel()
    channel2 = cv2.resize(img[:,:,1], size).ravel()
    channel3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((channel1, channel2, channel3))

def color_hist(img, nbins=32):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    hist_features = np.concatenate([channel1_hist[0], channel2_hist[0], channel3_hist[1]])
    return hist_features

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

def stack_arr(arr):
    # Stacks 1-channel array into 3-channel array to allow plotting
    return np.stack((arr, arr,arr), axis=2)

def draw_labeled_bboxes(img, labels):

    for car_number in range(1, labels[1]+1):

        nonzero = (labels[0] == car_number).nonzero()

        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255),6)

    return img

def process_image(img_src, vis_heat=False):

    draw_img = np.copy(img_src)
    current_draw_img = np.copy(img_src)
    scales = [1,1.5]
    img_src = img_src.astype(np.float32)/255
    heat_map = np.zeros_like(img_src[:,:,0])

    img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2YCrCb)
    img_cropped = img_src[y_start_stop[0]:y_start_stop[1],:,:]

    for scale in scales:
        if scale != 1:
            imshape = img_cropped.shape
            img_cropped = cv2.resize(img_cropped, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = img_cropped[:,:,0]
        ch2 = img_cropped[:,:,1]
        ch3 = img_cropped[:,:,2]

        nxblocks = (ch1.shape[1]//pix_per_cell) - 1
        nyblocks = (ch1.shape[0]//pix_per_cell) - 1
        window = 64
        nblock_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2
        nxsteps = (nxblocks - nblock_per_window) // cells_per_step
        nysteps = (nyblocks - nblock_per_window) // cells_per_step

        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):

                test_features = []
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                subimg = cv2.resize(img_cropped[ytop:ytop+window, xleft:xleft+window], (64,64))
                spatial_features = bin_spatial(subimg, size=spatial_size)
                test_features.append(spatial_features)
                hist_features = color_hist(subimg, nbins=hist_bins)
                test_features.append(hist_features)

                hog_feat1 = hog1[ypos:ypos+nblock_per_window, xpos:xpos+nblock_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblock_per_window, xpos:xpos+nblock_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblock_per_window, xpos:xpos+nblock_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                test_features.append(hog_features)

                test_features = np.concatenate(test_features)
                test_features = X_scaler.transform(test_features)

                test_prediction = svc.decision_function(test_features)

                if test_prediction > 0.75:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(current_draw_img, (xbox_left,ytop_draw+y_start_stop[0]),(xbox_left+win_draw,ytop_draw+y_start_stop[0]+win_draw), color=(0,0,255), thickness=6)
                    heat_map[ytop_draw+y_start_stop[0]:ytop_draw+y_start_stop[0]+win_draw,xbox_left:xbox_left+win_draw] += 1

    heat_map[heat_map <= 1] = 0
    heat_maps.append(heat_map)
    summed_heat_maps = np.sum(heat_maps, axis=0)
    #summed_heat_maps[summed_heat_maps <= 5] = 0
    labels = label(summed_heat_maps)
    draw_img = draw_labeled_bboxes(draw_img, labels)

    if vis_heat == True:
        diagScreen = np.zeros((720, 2560, 3), dtype=np.uint8)
        diagScreen[0:720, 0:1280] = draw_img
        diagScreen[0:720, 1280:2560] = stack_arr(heat_map*255//heat_map.max())
        return diagScreen
    else:
        return draw_img

if __name__ == '__main__':

    if 1:
        cars = loadimages('./vehicles')
        noncars = loadimages('./non-vehicles')

        car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)

        notcar_features = extract_features(noncars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()

    output = 'out.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    out_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    out_clip.write_videofile(output, audio=False)
