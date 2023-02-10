import torch
import random
import librosa
import numpy as np
from datasets import load_dataset
from IPython.display import Audio
from audiodiffusion import AudioDiffusion
from audiodiffusion.mel import Mel
import torchaudio
import torchvision

print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())

device = "cuda" if torch.cuda.is_available() else "cpu"
# else "mps" if torch.backends.mps.is_available()
generator = torch.Generator(device=device)

print("done")
# https://github.com/archinetai/audio-diffusion-pytorch/discussions/14

model_id = "teticio/audio-diffusion-ddim-256"
audio_diffusion = AudioDiffusion(model_id=model_id)
mel: Mel = audio_diffusion.pipe.mel

target_sample_rate = 22050


def readAudioFile(path):
    print("Reading audio file: " + path)

    raw_audio, sample_rate = librosa.load(path, sr=target_sample_rate)
    mel.load_audio(raw_audio=raw_audio)
    image = mel.audio_slice_to_image(0)
    audio=mel.get_audio_slice(0)

    # image, (sample_rate, audio) = audio_diffusion.generate_spectrogram_and_audio_from_audio(source)
    print("Encoding audio file: " + path)
    noise = audio_diffusion.pipe.encode([image])
    return [image, audio, noise, sample_rate]

source = "./bucky.wav"
target = "./makeba.wav"

source_image, source_audio, source_noise, sample_rate = readAudioFile(source)
taget_image, target_audio, target_noise, _ = readAudioFile(target)

print("Processing ...")

alpha = 0.5  #@param {type:"slider", min:0, max:1, step:0.1}
_, (sample_rate, audio) = audio_diffusion.generate_spectrogram_and_audio(
    noise=audio_diffusion.pipe.slerp(
        torch.tensor(source_noise),
        torch.tensor(target_noise),
        alpha),
    generator=generator)

print("Done. Saving audio file.")
torchaudio.save("output.wav", audio, sample_rate)
# source = readAudioFile("bucky.wav")
# target = readAudioFile("makeba.wav")

# N = source.shape[2]

# assert N == target.shape[2]
# print("Processing soure and target of of number of samples: " + str(N))
# print("input shape: " + str(source.shape))
