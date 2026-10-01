from dataclasses import dataclass

import numpy as np
import onnxruntime as ort


def as_ndarray(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Expected ndarray from ONNX session, got {type(value).__name__}")
    return value


@dataclass
class VadState:
    session: ort.InferenceSession
    threshold: float
    sample_rate: np.ndarray
    window: int
    context_size: int
    state: np.ndarray
    context: np.ndarray
    buffer: np.ndarray


def reset_vad(vad: VadState) -> None:
    vad.state = np.zeros((2, 1, 128), dtype=np.float32)
    vad.context = np.zeros((1, vad.context_size), dtype=np.float32)
    vad.buffer = np.array([], dtype=np.float32)


def hears_speech(vad: VadState, speech: np.ndarray) -> bool:
    vad.buffer = np.concatenate((vad.buffer, speech))
    speech_detected = False

    while len(vad.buffer) >= vad.window:
        chunk = vad.buffer[: vad.window].reshape(1, -1)
        vad.buffer = vad.buffer[vad.window :]
        model_input = np.concatenate((vad.context, chunk), axis=1).astype(np.float32)
        outputs = vad.session.run(
            None,
            {"input": model_input, "sr": vad.sample_rate, "state": vad.state},
        )
        probability = as_ndarray(outputs[0])
        vad.state = as_ndarray(outputs[1])
        vad.context = chunk[:, -vad.context_size :]
        if float(probability[0][0]) > vad.threshold:
            speech_detected = True

    return speech_detected


def voice_embedding(session: ort.InferenceSession, audio: np.ndarray) -> np.ndarray:
    first_output = as_ndarray(session.run(None, {"audio": audio})[0])
    return np.asarray(first_output[0])
