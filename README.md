# nn
Program which uses a trained neural network model to classify mouth positions in webcam video frames as open or closed

This project involved the following steps:
- Create image dataset (~4,000 images) for the open and closed mouth positions
- Use the data to train a neural network that is capable of classifying such images
- Export the trained model
- Import the model into an OpenCV-based Go application which continually feeds each active webcam frame to the model and produces a classification result

I created the dataset through individual screenshots of videos as well as extracted video frames. I used ffmpeg to split the frames of recorded video files and rename each file according to the class (mouth-position) which it represents. E.g. The filename of an image representing an open mouth would be labeled similar to "ope.123.png".

To train the model, I used tflearn and tensorflow (among other technologies) in a python-based application. I created a Docker container and executed the training application with varying hyperparameters until I found one with acceptable performance. Using a borrowed script (see powerpoint for link) I froze the model and exported it.

The second application is written in Go and makes use of Golang bindings for OpenCV. Each frame of a webcam stream is extracted and fed through the imported model, classifying the image. The classification/prediction is then printed at the top of the window.

Here is a shortlist of the utilized/relevant technologies:
- Python
- Golang
- GoCV
- Tensorboard
- Tflearn
- Tensorflow
- GoLand
- Github
- Docker
- ffmpeg
- PyCharm
- Sublime
