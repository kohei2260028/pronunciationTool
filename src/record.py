import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

from dotenv import load_dotenv
load_dotenv()

def record_audio(filename, duration=2, fs=16000):
    print(f"Recording: {filename}")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    write(filename, fs, audio)