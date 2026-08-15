import numpy as np
import soundfile as sf
import features.mfcc as mfcc


data, samplerate = sf.read('C:/Users/tommy/Desktop/ProgettiGit/audio-features-cpp/samples/go.wav', dtype='int16')

if(samplerate != 16000):
    raise ValueError(f"expected 16000 Hz, got {samplerate}")


features_py = mfcc.extract_mfcc(data)
print("shape:", features_py.shape)   # (61, 13)

result_cpp = np.loadtxt('C:/Users/tommy/Desktop/ProgettiGit/audio-features-cpp/go_cpp.csv', delimiter=",")

if features_py.shape != result_cpp.shape:
    raise ValueError(f"shape mismatch: {features_py.shape} vs {result_cpp.shape}")

diff = np.abs(features_py - result_cpp)

print("Differenza massima assoluta: ", np.max(diff))
print("Differenza media: ", np.mean(diff))
print("Dove sta il massimo: ", np.unravel_index(np.argmax(diff), diff.shape))

for k in range(mfcc.NUM_COEFFS):
    print(f"  c{k:<2d}: max diff = {np.max(diff[:, k]):.6f}")