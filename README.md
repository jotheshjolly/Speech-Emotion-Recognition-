# Speech-Emotion-Recognition-
classifies the emotion based on the user audio input
## Inspiration
A lot of things happening on earth but everyone is having their own problem for that we hear music which help better but there is tiny problem the music which we hear is not suggested by our emotion
hence we need a person to analyse it but not anymore here is our program to solve this problem
## What it does
it uses RAVDESS dataset to create a model using neural network
once the model is trained we can classify the live audio and play accordingly the emotion is.
we have 0.89p of the training data which is more enough to classify emotions 
## How we built it
we built with RAVDESS and TESS dataset
RAVDESS https://zenodo.org/record/1188976#.XsAXemgzaUk
TESS https://tspace.library.utoronto.ca/handle/1807/24487
here it has 8 classes which will give the emotions (0 = neutral, 1 = calm, 2 = happy, 3 = sad, 4 = angry, 5 = fearful, 6 = disgust, 7 = surprised)
 03-01-01-01-01-02-05.wav is an example of WRONG prediction
other audio is for testing the model
## Challenges we ran into
The RAVDESS contains 7356 files. Each file was rated 10 times on emotional validity, intensity, and genuineness.  training of huge data takes time and we uses model from the pre trained dataset for time being. the audio processing takes time and also due the system requirement we try to run in colab and replit so it takes lot of time to process it.
## Accomplishments that we're proud of
with help of the model we can able to classify the emotion in the users speech 
## What we learned
learned about keras and scipy and little about tensorflow 
## What's next for SPEECH EMOTIONAL RECOGNITION
building ui and get work with web framework using django and ask for api construction.
