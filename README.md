# Face-Mask-Detection-Web-App
>***This is only for education Purpose*** 

> About Project

This project uses a Deep Neural Network, more specifically a Convolutional Neural Network, to differentiate between images of people with and without masks. The CNN manages to get an accuracy of 97.6% on the training set and 97.3% on the test set. Then the stored weights of this CNN are used to classify as mask or no mask, in real time, using OpenCV. With the webcam capturing the video, the frames are preprocessed and and fed to the model to accomplish this task. The model works efficiently with no apparent lag time between wearing/removing mask and display of prediction.


## Technologies

* Keras/Tensorflow
* OpenCV
* Flask
* Numpy
* HTML & CSS

This whole Project is Deployed using Flask

## Steps used for Deployment:

a. Created the Deep learning model and saved it.

b. Created app.py and the webpage page it will render to.
