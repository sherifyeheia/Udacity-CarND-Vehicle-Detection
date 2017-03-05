##Vehicle Detection Project



####Sherif Mohamed

---

Note: I have extensivly used the code from the exampkes and quizes

[//]: # (Image References)

[image1]: ./output_test_images/car_test_img.jpg
[image2]: ./output_test_images/notcar_test_img.jpg
[image3]: ./output_test_images/car_test_img_hogimg_ch0.jpg
[image4]: ./output_test_images/car_test_img_hogimg_ch1.jpg
[image5]: ./output_test_images/car_test_img_hogimg_ch2.jpg
[image6]: ./output_test_images/notcar_test_img_hogimg_ch0.jpg
[image7]: ./output_test_images/notcar_test_img_hogimg_ch1.jpg
[image8]: ./output_test_images/notcar_test_img_hogimg_ch2.jpg
[image9]: ./output_test_images/window_img0.jpg
[image10]: ./output_test_images/window_img1.jpg
[image11]: ./output_test_images/window_img2.jpg
[image12]: ./output_test_images/all_window_img.jpg
[image13]: ./output_test_images/image_example_with_boxes1.jpg
[image14]: ./output_test_images/image_example_with_boxes6.jpg
[image15]: ./output_test_images/test_imgs_with_boxes_1.jpg
[image16]: ./output_test_images/test_heatmap_1.jpg
[image17]: ./output_test_images/test_imgs_with_boxes_2.jpg
[image18]: ./output_test_images/test_heatmap_2.jpg
[image19]: ./output_test_images/test_imgs_with_boxes_3.jpg
[image20]: ./output_test_images/test_heatmap_3.jpg
[image21]: ./output_test_images/test_imgs_with_boxes_4.jpg
[image22]: ./output_test_images/test_heatmap_4.jpg
[image23]: ./output_test_images/test_imgs_with_boxes_5.jpg
[image24]: ./output_test_images/test_heatmap_5.jpg
[image25]: ./output_test_images/test_imgs_with_boxes_6.jpg
[image26]: ./output_test_images/test_heatmap_6.jpg
[image27]: ./examples/HOG_example.jpg


[video1]: ./output_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points


---

###Histogram of Oriented Gradients (HOG)
####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

#####++a- Reading all Training Images (Cars, Not Cars) using **glob**++

Cars    :**  8792**

Notcars **8968**

```
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

```

| Car Image Example | NotCar Image Example |
|--------|--------|
|    ![alt text][image1]     |    ![alt text][image2]     |


#####++b- Feature Extraction for both Cars and Nocars Images++

After Exploring different values and looking what other colleagurés best practices.
I have extracted only HOG features with the following parameters.


```
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

```
This is an example for the previous two images

| Car Channel 0 HOG | Car Channel 1 HOG |Car  Channel 2 HOG |
|--------|--------|
|    ![alt text][image3]     |  ![alt text][image4]       |   ![alt text][image5]     |






| NotCar Channel 0 HOG | NotCar Channel 1 HOG |NotCar Channel 2 HOG |
|--------|--------|
|     ![alt text][image6]      |  ![alt text][image7]        |   ![alt text][image8]    |


####2. Explain how you settled on your final choice of HOG parameters.

I have tried several cominations of HOG parameters and searched also what other colleagues have done and i used these parameterrs giving good performance.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I have trained the classifier using the following steps:

**A- **The  features extracted is normalized

```
#%% Normalize features
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```
**B- **Labels Vector added and then it's shuffled randomly and splited into Training and Testing Features

```
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
```

**C-** rbf kernel SVM Classifier is used and then trained and then tested with the test features
**Accuracy =  0.989 **

```
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
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

**a- Sliding Windows definition.**
I have defined three windows configuration to cover the area of interest.
Smaller at the horizon and bigger close to the car.

I defined a function **get_frame_hotwindows** which takes the image and search using the defined windows and returns with the hot widows with possible cars.
```
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
```
| First Windows | 
|--------|
|      ![alt text][image9]   | 

|  Second Windows  | 
|--------|
|      ![alt text][image10]   | 

|  Third Windows  | 
|--------|
|      ![alt text][image11]   | 

| Combined together | 
|--------|
|      ![alt text][image12]   | 


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?


|  |  |
|--------|--------|
|    ![alt text][image13]     |   ![alt text][image14]       |


---


### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)


Here's a [link to my video result](./project_video.mp4)
https://youtu.be/MHp400g3ESg

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.



- **average_over_frames** is used to get all the hot windows over the last 10 frames.

- **add_heat** is used to add values to the heat map corresponding to the hot windows detected.

- **apply_threshold** to apply a threshold to the heat map to eleminate false positives.

- **grey_opening**  is used to smooth the windos detected to elminate the presense of redundant car detection for the same car.

- **label** is used to give lables for the detected boxes from the hetmap.

- **draw_labeled_bboxes** is used to draw the final box using the information from the detected box.

The boundaries of the final box is detected from the min and max values of the detected boxes of the same car lable.

```    
	frame_hot_windows = get_frame_hotwindows(image)
    avg_hot_windows = average_over_frames(frame_hot_windows)
    heat = add_heat(heat,avg_hot_windows)   
    threshold = np.mean(heat[heat>0])
    heat = apply_threshold(heat,threshold)
    heat = grey_opening(heat, size=(45,45))
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(image_orig, labels)

```

### Here are six frames and their corresponding heatmaps:

|    ![alt text][image15]       ![alt text][image16]       



|    ![alt text][image17]       ![alt text][image18]       



|    ![alt text][image19]       ![alt text][image20]       



|    ![alt text][image21]      ![alt text][image22]       



|    ![alt text][image23]       ![alt text][image24]       



|    ![alt text][image25]       ![alt text][image26]       



---


###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems I had :
- **Redundant labling for the same car **.

  I have tried to solve this problem by smoothing the detected boxes using **grey_opening**

  I would propose to make an algorith to give different weights for the detected windows.

  So the windos closer to the previous right detection has higher values in the heatmap other than windows found othe place.













