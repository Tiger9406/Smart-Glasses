import wave

import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("redimnet_b2.onnx")


def load_wav(path):
    with wave.open(path, "rb") as f:
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return (audio.astype(np.float32) / 32767.0).reshape(1, -1)


def get_embedding(audio):
    return session.run(None, {"audio": audio})[0][0]


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Get embeddings
emb1 = get_embedding(load_wav("ShaunTestOne.wav"))  # you speaking
emb2 = get_embedding(load_wav("ShaunTestTwo.wav"))  # you speaking again
emb3 = get_embedding(load_wav("TigerTestOne.wav"))  # someone else

# Compare
print(f"You vs You:    {cosine_sim(emb1, emb2):.3f}")  # should be high ~0.7+
print(
    f"You vs Other:  {cosine_sim(emb1, emb3):.3f}"
)  # should be low ~0.3 or lessprint(f"You vs Other:  {cosine_sim(emb1, emb3):.3f}")   # should be low ~0.3 or less
