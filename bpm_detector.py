import librosa

#source = "./beat1-124.wav"
# source = "./beat2-123.wav"
# source = "./mix1-127.wav"
# source = "./bass1-127.wav"

#source = "./mix2-122.wav"
#source = "./drum3-122.wav"

import sys
print (sys.argv)
source = sys.argv[1]
if source is None:
    source = "./piano1-122.wav"

source_audio,  sample_rate = librosa.load(source)
bpm, beats = librosa.beat.beat_track(y=source_audio, sr=sample_rate)
tempo = librosa.beat.tempo(y=source_audio, sr=sample_rate)[0]
print(source + ", tempo:" + str(tempo) + " bpm:" + str(bpm))
print("Beats:" , beats)
print("done")
