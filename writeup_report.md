# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./output_images/000.jpg 
[image1]: ./output_images/001.jpg 
[image2]: ./output_images/002.jpg
[image3]: ./output_images/003.jpg
[image4]: ./output_images/004.jpg
[image5]: ./output_images/005.jpg 
[image6]: ./output_images/006.jpg 
[image7]: ./output_images/007.jpg 
[image8]: ./output_images/008.jpg 
[image9]: ./output_images/009.jpg 
[image10]: ./output_images/010.jpg 
[image11]: ./output_images/011.jpg 
[image12]: ./output_images/012.jpg 
[image13]: ./output_images/013.jpg 
[image14]: ./output_images/014.jpg 

[image15]: ./output_images/All_Signs.png "All_Signs"
[image16]: ./output_images/Histogram.png "Histogram"
[image17]: ./output_images/Grayscale.png "Grayscale"
[image18]: ./output_images/Normalization.png "Normalization"
[image19]: ./images/resized/resized_images.png "webimages"
[image20]: ./output_images/softmax_prob.png "Softmax_Probability"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jkstyle2/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is "34799"
* The size of the validation set is "4410"
* The size of test set is "12630"
* The shape of a traffic sign image is "(32, 32, 3)"
* The number of unique classes/labels in the data set is "43"

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
The training images are randomly chosen as below:
![All_Signs][image15]

It is a bar chart showing how the labels are distributed in each data set:
![Histogram][image16]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the grayscale images are much faster to process rather than color images, as it has only one channel/depth, so that we can speed up learning our model.
Color information might be used effectively, but in real-world system the information is often easily corrupted with some illumination conditions such as sunlight, headlights, neon/LED signs, traffic lights. For the reason, the colored inputs should be augmented with "incorrectly colored" or "artificaially tinted" inputs.

Here is an example of a traffic sign image before and after grayscaling.
![Grayscale][image17]


As a second step, I normalized the image data because it can speed up convergence and the neural networks can perform better when the input(feature) distributions have mean zero. As a suggested way, I used a technique such that (X_train - 128)/128 for the normalization, which results in not exactly zero mean, but fairly easy to implement. It seems worked great but I still wonder how this technique really made it although it doesn't use the exact mean and variance. 
The values are all reduced to the range (-1,1) and the resulting outputs are as below.

![Normalization][image18]


For the last step, It is obvious to generate additional data because the number of images in some classes is relatively small. The augmented data set adds extra value to base data so that we can take more abundant information to classify the dataset. Due to the shortage of time, I just skip this step for now, but I will be back to complete this task later when I get more time.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

 
<table>
    <tr>
        <th>STEP</th>
        <th>LAYER</th>
        <th>IN. SIZE</th>
        <th>OUT. SIZE</th>
        <th>DESCRIPTION</th>
    </tr>
    <tr>
        <td>Input image</td>
        <td></td>
        <td>32 × 32 × 1</td>
        <td>32 × 32 × 1</td>
        <td>32 × 32 × 1 grayscale image.</td>
    </tr>
    <tr>
        <td rowspan="4">Convolution 1</td>
        <td>Convolution 5 × 5</td>
        <td>32 × 32 × 1</td>
        <td>28 × 28 × 16</td>
        <td>1 × 1 stride, VALID padding</td>
    </tr>
    <tr>
        <td>RELU</td>
        <td>28 × 28 × 16</td>
        <td>28 × 28 × 16</td>
        <td></td>
    </tr>
    <tr>
        <td>Max pooling</td>
        <td>28 × 28 × 16</td>
        <td>14 × 14 × 16</td>
        <td>2 × 2 stride, VALID padding</td>
    </tr>
    <tr>
        <td>Dropout</td>
        <td>14 × 14 × 16</td>
        <td>14 × 14 × 16</td>
        <td>0.8 keep rate</td>
    </tr>
    <tr>
        <td rowspan="4">Convolution 2</td>
        <td>Convolution 5 × 5</td>
        <td>14 × 14 × 16</td>
        <td>10 × 10 × 32</td>
        <td>1 × 1 stride, SAME padding</td>
    </tr>
    <tr>
        <td>RELU</td>
        <td>10 × 10 × 32</td>
        <td>10 × 10 × 32</td>
        <td></td>
    </tr>
    <tr>
        <td>Max pooling</td>
        <td>10 × 10 × 32</td>
        <td>5 × 5 × 32</td>
        <td>2 × 2 stride, VALID padding</td>
    </tr>
    <tr>
        <td>Dropout</td>
        <td>5 × 5 × 32</td>
        <td>5 × 5 × 32</td>
        <td>0.8 keep rate</td>
    </tr>
    <tr>
        <td>Flattening</td>
        <td>Flatten</td>
        <td>5 × 5 × 32</td>
        <td>1 × 800</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">Fully Connected 1</td>
        <td>Fully connected</td>
        <td>1 × 800</td>
        <td>1 × 256</td>
        <td></td>
    </tr>
    <tr>
        <td>RELU</td>
        <td>1 × 256</td>
        <td>1 × 256</td>
        <td></td>
    </tr>
    <tr>
        <td>Dropout</td>
        <td>1 × 256</td>
        <td>1 × 256</td>
        <td>0.8 keep rate</td>
    </tr>
    <tr>
        <td rowspan="2">Fully Connected 2</td>
        <td>Fully connected</td>
        <td>1 × 256</td>
        <td>1 × 128</td>
        <td></td>
    </tr>
    <tr>
        <td>Dropout</td>
        <td>1 × 128</td>
        <td>1 × 128</td>
        <td>0.8 keep rate</td>
    </tr>
    <tr>
        <td>Fully Connected 3</td>
        <td>Fully connected</td>
        <td>1 × 128</td>
        <td>1 × 43</td>
        <td></td>
    </tr>
</table>


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

There are several popular ways to use optimization techniques: SGD, SGD+momentum, Adagrad, Adadelta and Adam - methods for finding local optimum (global when dealing with convex problem) of certain differentiable functions. 
Through the lecture, I learnt about SGD using mini-batch and its mechanism how it effectively works is quite understandable compared to full-batch Gradient descent method. After that, in the LeNet Lab section, Adam optimizer was suddenly introduced with very brief explanation, and I just have used it without any clear understanding.

After taking a look at [this comparison of optimizers](http://int8.io/comparison-of-optimization-techniques-stochastic-gradient-descent-momentum-adagrad-and-adadelta/#AdaGrad_8211_experiments), now I could get some knowledge about Adam, and I decided to use it as an optimizer. With RMSProp and Momentum, the optimizer can find a proper direction and a step size.

I kept tring to find out the optimum value of hyperparameters such as learning rate, dropout probability and batch size depending on its fitting condition. 
The results are as below:

| EPOCHS | LEARNING RATE | BATCH SIZE | KEEP PROB  | VAL. ACC. | TEST ACC. | CONCLUSION / ACTION |
|-------------------|-----------------------------------|---------------------------|---------------------------|----------------------|------------------------|-------------------------------------------------|
|     10     |          0.05         |        64        |       0.5       |    0.048   |     0.100    | Underfit. Adjust params.
|     10     |          0.05         |       128       |       0.5       |    0.054   |     0.050    | Underfit. Adjust params.
|     10     |          0.05         |        64        |       0.7       |    0.054   |     0.050    | Underfit. Adjust params.
|     10     |          0.001       |        64        |       0.5       |    0.926   |     0.905    | Overfit. Keep Learning rate
|     20     |          0.001       |        64        |       0.5       |    0.935   |     0.928    | Overfit. Adjust params.
|     10     |          0.001       |       128       |       0.5       |    0.932   |     0.911    | Overfit. Adjust params.
|     10     |          0.001       |       128       |       0.6       |    0.918   |     0.918    | Underfit. Adjust params.
|     10     |          0.001       |       128       |       0.8       |    0.934   |     0.928    | Overfit. Adjust params.
|     20     |          0.001       |       128       |       0.8       |    0.928   |     0.928    | Underfit. Adjust params.
|     20     |          0.001       |       128       |       0.7       |    0.954   |     0.930    | Overfit. Adjust params.
|     20     |          0.001       |       128       |       0.6       |    0.955   |     0.928    | Overfit. Adjust params.
|     20     |          0.001       |       128       |       0.5       |    0.944   |     0.919    | Overfit. Adjust params.
|     20     |          0.001       |       128       |       0.55     |    0.939   |     0.919    | Overfit. Adjust params.
|     30     |          0.001       |       128       |       0.5       |    0.952   |     0.928    | Overfit. Adjust params.
|     30     |          0.001       |       256       |       0.5       |    0.937   |     0.924    | Underfit. Adjust params.
|     30     |          0.001       |       128       |       0.6       |    0.947   |     0.929    | Underfit. Adjust params.
|     30     |          0.001       |       128       |       0.7       |    0.950   |     0.928    | Looks better. Increase epochs
|     60     |          0.001       |       128       |       0.7       |    0.955   |     0.940    | 


The final parameter settings used are:
- epochs: 60
- learning rate: 0.001
- batch size: 128
- dropout keep probability: 0.7
- mu: 0
- sigma: 0.1


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results are:
* training set accuracy of 100%
* validation set accuracy of 95.5% 
* test set accuracy of 94.0%



If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?



I first implemented the same architecture from the model I learnt in the LeNet Lab. It went well with over 85% validation accuracy, but still needed some modification to reach out to over 93% validation accuracy. 
To overcome the underfitting issue, I mainly tuned some parameters such that making output channels much deeper, and also added dropout function in each Fully-connected layer. Initially parameters were just guessed and then continuously adjusted by trial and error. 
The adjusted architecture had some overfitting issues, so that I had to try using dropout layers by spending some time fine-tunning its parameters. By using dropout layers, the dependency of training dataset could be minimized, so I could solve the overfitting issue quite effectively.

While mostly tuning the parameters, I felt that I need to figure out further about the mechanism of deep neural network, so that I could minimize spending hours to find the best suitable parameters.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some German traffic signs that I found on the web:

![webimages][image19]

I guess the second image might be difficult to classify because it seems zoomed in too much. It's better to have somewhat margins around the signs to recognize the images well. What's worse, the second image "STOP" has fewer training dataset compared to others, so that it would be one of the tricky images to be classified.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (30km/h)      		| Speed Limit (30km/h)   									| 
| Stop     			| Bumpy road 										|
| Priority road					| Priority road											|
| Road work	      		| Road work					 				|
| Yield			| Yield

4 out of 5 images were classified correctly. It indicates the network has accuracy of 80% on these images, and it may not compare favorably to the accuracy on the test set of 92.8%. 
As the original image 'Stop' was predicted as a 'Bumpy road' image, I also suspect the model has trouble prediction on 'Bumpy road' images.
I consider that further augmenting techniques including rotation, translation, zoom, flip would improve the model's performance well.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a Speed Limit (30km/h) sign (probability of 0.99), and the image does contain a speed limit sign. The rest of them have similar certainty except the 2nd image 'Stop' sign. The 'Stop' sign image is misclassified as expected. 
The top five soft max probabilities on each image are :

![Softmax_Probability][image20]




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

- I'll do the task later on.

