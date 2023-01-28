
from audio_diffusion_pytorch import AudioDiffusionModel
import torch
import torchaudio

# https://github.com/archinetai/audio-diffusion-pytorch/discussions/14

model = AudioDiffusionModel(
    in_channels=1,
    context_channels=[1]
)


def reshape(tensor):
    assert tensor.shape[0] == 2
    return tensor[0, :].unsqueeze(0).unsqueeze(0)  # tensor.shape = [1, 1, N]


def readAudioFile(filename):
    [rawSource, sampleRate] = torchaudio.load(
        filename)  # rawSource.shape = [2, N]
    return reshape(rawSource)


source = readAudioFile("bucky.wav")
target = readAudioFile("makeba.wav")

N = source.shape[2]
print("Processing soure and target of of number of samples: " + str(N))
print("input shape: " + str(source.shape))

# Train model with pairs of audio sources, i.e. predict target given source
# [batch, in_channels, samples], 2**18 ≈ 12s of audio at a frequency of 22050

# loss = model(target, channels_list=[source])
loss = model(target, context=[source])
loss.backward()  # Do this many times

# Sample a target audio given start noise and source audio
noise = torch.randn(1, 1, N)

print("Starting sampling...")
sampled = model.sample(
    context=[source],
    noise=noise,
    num_steps=25  # Suggested range: 2-50
)  # [2, 1, 2 ** 18]

print("Saving output.wav")
torchaudio.save("output.wav", sampled[0], 48000)
