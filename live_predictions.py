import keras
import librosa
import numpy as np
import sys
import pathlib
import subprocess
import sounddevice as sd
from scipy.io.wavfile import write

working_dir_path = pathlib.Path().absolute()
if sys.platform.startswith('win32'):
    MODEL_DIR_PATH = str(working_dir_path) + '\\model\\'
    EXAMPLES_PATH = str(working_dir_path) + '\\examples\\'
else:
    MODEL_DIR_PATH = str(working_dir_path) + '/model/'
    EXAMPLES_PATH = str(working_dir_path) + '/examples/'
    

duration = 3.5
freq = 44400

print("recording started")
recording = sd.rec(int(duration * freq),samplerate = freq, channels = 2)
# Wait for the audio to complete
sd.wait()
write(EXAMPLES_PATH+"recording0.wav", freq, recording)
print("recording ended")

class LivePredictions:
    """
    Main class of the application.
    """

    def __init__(self, file):
        self.file = file
        self.path = MODEL_DIR_PATH + 'Emotion_Voice_Detection_Model.h5'
        self.loaded_model = keras.models.load_model(self.path)

    def make_predictions(self):
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        #x = np.expand_dims(mfccs, axis=2)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        print( "Prediction is", " ", self.convert_class_to_emotion(predictions))
        return self.convert_class_to_emotion(predictions)

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label

def play_sound(apath):
    subprocess.call(["afplay", EXAMPLES_PATH + apath])
    

if __name__ == '__main__':
    print("prediction started")
    live_prediction = LivePredictions(file=EXAMPLES_PATH + 'recording0.wav')
    live_prediction.loaded_model.summary()
    p = live_prediction.make_predictions()
    
    if p == 'calm':
        print("playing calm")
        play_sound("calm.mp3")
    elif p == 'happy':
        print("playing happy")
        play_sound("happy.mp3")
    elif p == 'sad':
        print("playing sad")
        play_sound("sad.wav")
    elif p == 'angry':
        print("playing angry")
        play_sound("angry.mp3")
    elif p == 'fearful':
        print("playing fearful")
        play_sound("fear.mp3")
    elif p == 'disgust':
        print("playing disgust")
        play_sound("disgust.mp3")
    elif p == 'surprised':
        print("playing suprised")
        play_sound("suprise.mp3")

        
