# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

# Sampling frequency
freq = 44400

# Recording duration in seconds
duration = 3.5

# to record audio from
# sound-device into a Numpy
print("recording started")
recording = sd.rec(int(duration * freq),
				samplerate = freq, channels = 2)

# Wait for the audio to complete
sd.wait()

# using scipy to save the recording in .wav format
# This will convert the NumPy array
# to an audio file with the given sampling frequency
write("recording0.wav", freq, recording)

# using wavio to save the recording in .wav format
# This will convert the NumPy array to an audio
# file with the given sampling frequency
wv.write("recording1.wav", recording, freq, sampwidth=2)
