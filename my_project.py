# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:14:35 2017

@author: sherif Moahmed
"""


import cv2
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from moviepy.editor import VideoFileClip
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import grey_opening

#%%

  
def bin_spatial(img, size=(32, 32)):
        """ compute binned color features  
    Args:
        image (numpy.array): image,new size
    Returns:
        feature vector
    """

    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features
#%%

# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
            """  histogram of the color channels separately
    Args:
        image (numpy.array): image,bins_range
    Returns:
        feature vector
    """

    
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

#%%

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    
    """  compute HOG features and visualization
    Args:
        image (numpy.array): image,bins_range
    Returns:
        feature vector
    """

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
#%%



# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """  o extract features from a list of images
    Args:
        image (numpy.array): image, Different feature parameters
    Returns:
        feature vector
    """   
    
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
    
    
#%%

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True): 
    

    """  extract features from a single image window
           This function is very similar to extract_features()
         just for a single image rather than list of images
    Args:
        image (numpy.array): image, Different feature parameters
    Returns:
        feature vector
    """   
    
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
#%%

def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    
    
    """  search for hot windows possible to be cars by predicring 
        using the previously trained classifier
    Args:
        image (numpy.array): image, list of windows to be searched for hot windows
    Returns:
        hot windows possible cars
    """   

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

#%%


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    
    """  draw bounding boxes
    Args:
        image (numpy.array): image, list of boxes to be drawn,color,line thickness
    Returns:
        image with the boxes drawn
    """   
    
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

#%%

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    
        """  creat windows to be used later for searching
    Args:
        image (numpy.array): image, X start and stop positions, Y start and stop positions,
        Windows size, Windows overlap
    Returns:
        list of windows
    """   
    
    
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


#%%
### TODO: Tweak these parameters and see how the results change.
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 1 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
#y_start_stop = [None, None] # Min and max in y to search in slide_window()



#%%Read training  data
# loading project dataset
# car images and none car images

# get vehicles and non-vehicles images from here
# https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
# https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip
# and extract into dataset directory
cars = glob.glob('training-data/vehicles/**/*.png', recursive=True)
notcars = glob.glob('training-data/non-vehicles/**/*.png', recursive=True)

# cloading car images
cars_img = []
for impath in cars:
    cars_img.append (mpimg.imread(impath))

# loading non car images
notcars_img = []
for impath in notcars:
    notcars_img.append (mpimg.imread(impath))

car_image_count = len (cars_img)
notcar_image_count = len (notcars_img)

print ('dataset has cars:', car_image_count)
print ('none cars:', notcar_image_count)

#%%

"""
This is used for testing purposes
"""
car_test_img = cars_img[20]
notcar_test_img = notcars_img[20]

car_test_img_feat_ch0,car_test_img_hogimg_ch0 = get_hog_features(car_test_img[:,:,0], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=True, feature_vec=True)

car_test_img_feat_ch1,car_test_img_hogimg_ch1 = get_hog_features(car_test_img[:,:,1], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=True, feature_vec=True)

car_test_img_feat_ch2,car_test_img_hogimg_ch2 = get_hog_features(car_test_img[:,:,2], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=True, feature_vec=True)

notcar_test_img_feat_ch0,notcar_test_img_hogimg_ch0 = get_hog_features(notcar_test_img[:,:,0], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=True, feature_vec=True)

notcar_test_img_feat_ch1,notcar_test_img_hogimg_ch1 = get_hog_features(notcar_test_img[:,:,1], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=True, feature_vec=True)

notcar_test_img_feat_ch2,notcar_test_img_hogimg_ch2 = get_hog_features(notcar_test_img[:,:,2], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=True, feature_vec=True)


mpimg.imsave( "output_test_images/car_test_img_hogimg_ch0.jpg", car_test_img_hogimg_ch0 )
mpimg.imsave( "output_test_images/car_test_img_hogimg_ch1.jpg", car_test_img_hogimg_ch1 )
mpimg.imsave( "output_test_images/car_test_img_hogimg_ch2.jpg", car_test_img_hogimg_ch2 )

mpimg.imsave( "output_test_images/notcar_test_img_hogimg_ch0.jpg", notcar_test_img_hogimg_ch0 )
mpimg.imsave( "output_test_images/notcar_test_img_hogimg_ch1.jpg", notcar_test_img_hogimg_ch1 )
mpimg.imsave( "output_test_images/notcar_test_img_hogimg_ch2.jpg", notcar_test_img_hogimg_ch2 )
#mpimg.imsave( "output_test_images/notcar_test_img.jpg", notcar_test_img )



#%% 
"""
extract features from the training data
"""
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)


#%% Normalize features
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
#%%

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

#%%Train the Classifier

#  rbf kernel SVM
svc = SVC ()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()




#%%
def draw_labeled_bboxes(img, labels):
        
    """ draw the labeled boxes from the heatmap after averaging them to get the final bounding box
    Args:
        image (numpy.array): image, Labels found in the heatmap
    Returns:
        image with Final bouding boxes drawn
    """   
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


#%%

def apply_threshold(heatmap, threshold):
        """ Apply threshold to the heatmap to remove false positives
    Args:
        image (numpy.array): image, Threshold Value
    Returns:
        Heatmap after applying the threshold
    """ 
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


#%%

def add_heat(heatmap, bbox_list):
    """ add values to the heatmap for each pixel found inside the bounding boxes
    Args:
        image (numpy.array): image, list of boxes
    Returns:
        Heatmap with values added for boxes found
    """ 
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes


#%%  dfeine three widows to search 

sw_x_limits = [
    [None, None],
    [32, None],
    [0, 1280]
]

sw_y_limits = [
    [400, 700],
    [400, 600],
    [390, 540]
]

sw_window_size = [
    (128, 128),
    (96, 96),
    (80, 80)
]

sw_overlap = [
    (0.5, 0.5),
    (0.5, 0.5),
    (0.5, 0.5)
]

def get_frame_hotwindows(image):
    """ define the windows to be used for search and extract the hot windows from the input image
    Args:
        image (numpy.array): image
    Returns:
        list of hot windows
    """ 
    frame_hot_windows = []
    # create sliding windows
    windows0 = slide_window(image, x_start_stop=sw_x_limits[0], y_start_stop=sw_y_limits[0], 
                        xy_window=sw_window_size[0], xy_overlap=sw_overlap[0])
    
    windows1 = slide_window(image, x_start_stop=sw_x_limits[1], y_start_stop=sw_y_limits[1], 
                        xy_window=sw_window_size[1], xy_overlap=sw_overlap[1])
    
    windows2 = slide_window(image, x_start_stop=sw_x_limits[2], y_start_stop=sw_y_limits[2], 
                        xy_window=sw_window_size[2], xy_overlap=sw_overlap[2])
    
    
    
    hot_windows = search_windows(image, windows0, svc, X_scaler, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
    
    hot_windows2 = search_windows(image, windows1, svc, X_scaler, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)  
            
    hot_windows3 = search_windows(image, windows2, svc, X_scaler, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)  

    frame_hot_windows.extend(hot_windows)
    frame_hot_windows.extend(hot_windows2)
    frame_hot_windows.extend(hot_windows3)
    
    return frame_hot_windows
#%%
"""
This is used for testing purposes
"""
image = mpimg.imread('test_images/test1.jpg')

windows = []
windows0 = slide_window(image, x_start_stop=sw_x_limits[0], y_start_stop=sw_y_limits[0], 
                    xy_window=sw_window_size[0], xy_overlap=sw_overlap[0])

windows1 = slide_window(image, x_start_stop=sw_x_limits[1], y_start_stop=sw_y_limits[1], 
                    xy_window=sw_window_size[1], xy_overlap=sw_overlap[1])

windows2 = slide_window(image, x_start_stop=sw_x_limits[2], y_start_stop=sw_y_limits[2], 
                        xy_window=sw_window_size[2], xy_overlap=sw_overlap[2])
 
windows.extend(windows0)
windows.extend(windows1)  
windows.extend(windows2)
window_img0 = draw_boxes(image, windows0, color=(0, 255, 255), thick=3)
window_img1 = draw_boxes(image, windows1, color=(0, 255, 255), thick=3)
window_img2 = draw_boxes(image, windows2, color=(0, 255, 255), thick=3)                        
all_window_img = draw_boxes(image, windows, color=(0, 255, 255), thick=3)

mpimg.imsave( "output_test_images/window_img0.jpg", window_img0 )
mpimg.imsave( "output_test_images/window_img1.jpg", window_img1 )
mpimg.imsave( "output_test_images/window_img2.jpg", window_img2 )
mpimg.imsave( "output_test_images/all_window_img.jpg", all_window_img )
                    
#plt.imshow(window_img)

#%%
class hot_boxes_class():
    def __init__(self):

            self.avg_hot_boxes = []
            self.frame_count = 0
        
    def re_init(self):
        self.__init__()


hot_boxes_class = hot_boxes_class()
#%%
def average_over_frames(frame_hot_windows):
        """ collest the hot windows over 10 frames to smooth the calculated heat map
    Args:
       the hot  windows found in the latest frame
    Returns:
        list of all  hot windows in the previous 10 frames
    """ 
    
    avg_hot_windows = []
    hot_boxes_class.frame_count = hot_boxes_class.frame_count +1
    
    if hot_boxes_class.frame_count < 10 :
        
        
        hot_boxes_class.avg_hot_boxes.append(frame_hot_windows)
        

    else:
        
        hot_boxes_class.avg_hot_boxes.pop(0)
        hot_boxes_class.avg_hot_boxes.append(frame_hot_windows)
        
        
    for boxes in hot_boxes_class.avg_hot_boxes:
        avg_hot_windows.extend(boxes)
        
    return avg_hot_windows
        
#%%
"""
This is used for testing purposes
"""
test_image = mpimg.imread('test_images/test1.jpg')
frame_hot_windows = get_frame_hotwindows(test_image)
image_example_with_boxes = draw_boxes(test_image, frame_hot_windows, color=(0, 255, 255), thick=3)
image_example_with_boxes= cv2.resize(image_example_with_boxes,(300,200))
mpimg.imsave( "output_test_images/image_example_with_boxes1.jpg", image_example_with_boxes )
plt.imshow(image_example_with_boxes) 
#%%   Test Code for averaging box and heatmaps
"""
This is used for testing purposes
"""
def test_pipeline(image):

    image_orig = np.copy (image)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    
    #image = image_orig.astype(np.float32)/255 
    frame_hot_windows = get_frame_hotwindows(image)
    #print(frame_hot_windows)
    #avg_hot_windows = average_over_frames(frame_hot_windows)
    
    # Add heat to each box in box list
    heat = add_heat(heat,frame_hot_windows)
    
    threshold = np.mean(heat[heat>0])
    heat = apply_threshold(heat,threshold)
    
    heat = grey_opening(heat, size=(45,45))
    
    # Apply threshold to help remove false positives
    
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(image_orig, labels)

    return draw_img,heatmap


#test_image = mpimg.imread('test_images/test1.jpg')
#test_imgs_with_boxes_1,test_heatmap_1 = test_pipeline(test_image)
#
#test_heatmap_1= cv2.resize(test_heatmap_1,(300,200))
#test_imgs_with_boxes_1 = cv2.resize(test_imgs_with_boxes_1,(300,200))
#
#mpimg.imsave( "output_test_images/test_heatmap_1.jpg", test_heatmap_1 )
#mpimg.imsave( "output_test_images/test_imgs_with_boxes_1.jpg", test_imgs_with_boxes_1 )
#
#

test_image = mpimg.imread('test_images/test6.jpg')
test_imgs_with_boxes_6,test_heatmap_6 = test_pipeline(test_image)

test_heatmap_6= cv2.resize(test_heatmap_6,(300,200))
test_imgs_with_boxes_6 = cv2.resize(test_imgs_with_boxes_6,(300,200))

mpimg.imsave( "output_test_images/test_heatmap_6.jpg", test_heatmap_6 )
mpimg.imsave( "output_test_images/test_imgs_with_boxes_6.jpg", test_imgs_with_boxes_6 )

    
#%%
def process_frame (image):
        """ Video frame processing
        1- hot windows extracted.
        2- added to the last 10 frames windows
        3-Heatmap is created for these windows
        4- Heatmap is thresholded to remove fals positives
        5- Heatmap is smoothed to remove redundant labling of the same car
        6- Labels is created for each possibl cars
        7- Final boxes drawn on the frame
    Args:
       frame to be processed
    Returns:
        image with boxes around possible cars
    """  
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    image_orig = np.copy (image)
    
    #image = image_orig.astype(np.float32)/255 
    frame_hot_windows = get_frame_hotwindows(image)
    
    avg_hot_windows = average_over_frames(frame_hot_windows)
    heat = add_heat(heat,avg_hot_windows)   
    
    threshold = np.mean(heat[heat>0])
    heat = apply_threshold(heat,threshold)
    
    heat = grey_opening(heat, size=(45,45))
    """
        # Add heat to each box in box list

    heat = grey_opening(heat, size=(5,5))
    
    #threshold = np.mean(heat[heat>10])
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,20)
    """
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(image_orig, labels)
    
    return draw_img



#%%
hot_boxes_class.re_init()

output_test_video = 'output_test_video.mp4'
clip1 = VideoFileClip('test_video.mp4')
#clip1 = clip1.subclip(t_start=36, t_end=50)
project_video_clip = clip1.fl_image(process_frame)
project_video_clip.write_videofile(output_test_video, audio=False)

#%%
hot_boxes_class.re_init()
output_project_video = 'output_project_video.mp4'
clip1 = VideoFileClip('project_video.mp4')
#clip1 = clip1.subclip(t_start=26, t_end=36)
project_video_clip = clip1.fl_image(process_frame)
project_video_clip.write_videofile(output_project_video, audio=False)



















