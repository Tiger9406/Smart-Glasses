"""Measure transcription real-time factor (RTF) for the audio pipeline.

Mirrors workers/audio.py exactly:
- 160 ms chunks at 16 kHz (core/config_audio.py)
- Silero VAD ONNX, 512-sample windows + 64-sample context, 0.5 threshold
- sentence ends after SILENT_CHUNK_THRESHOLD silent chunks, 320 ms pre-speech padding
- per sentence: parakeet transcribe_stream(context_size=(CONTEXT_LEFT, CONTEXT_RIGHT)),
  single add_audio() of the whole sentence, then read result.text
- per sentence: ReDimNet b2 ONNX speaker embedding on the same audio

RTF = processing_time / audio_duration (lower is better; < 1.0 means faster
than real time). Run from the repo root:

    python3 tests/benchmark_rtf.py [wav files...]
"""

import collections
import statistics
import sys
import time
import wave

import mlx.core as mx
import numpy as np
import onnxruntime as ort
from parakeet_mlx import from_pretrained

sys.path.insert(0, ".")
from core import config_audio  # noqa: E402

DEFAULT_WAVS = [
    "api/simulator_resources/shrish_introduced.wav",
    "api/simulator_resources/riley_self_intro.wav",
    "api/simulator_resources/recorded_audio.wav",
    "workers/audio_utils/solo_voices/Matt.wav",
    "workers/audio_utils/solo_voices/Shaun.wav",
    "workers/audio_utils/solo_voices/TigerGroupChat.wav",
    "workers/audio_utils/solo_voices/TigerTestOne.wav",
]

SAMPLE_RATE = config_audio.AUDIO_SAMPLE_RATE_HZ
CHUNK_SAMPLES = int(SAMPLE_RATE * config_audio.AUDIO_CHUNK_SIZE_MS / 1000)
PADDING_CHUNKS = int(320 / config_audio.AUDIO_CHUNK_SIZE_MS)


class VadSegmenter:
    """Port of AudioWorker's Silero VAD sentence segmentation (workers/audio.py)."""

    def __init__(self):
        self.session = ort.InferenceSession(config_audio.VAD_PATH)
        self.threshold = 0.5
        self.window = 512
        self.context_size = 64
        self.sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, self.context_size), dtype=np.float32)
        self.buffer = np.array([], dtype=np.float32)

    def is_speech(self, samples: np.ndarray) -> bool:
        self.buffer = np.concatenate((self.buffer, samples))
        detected = False
        while len(self.buffer) >= self.window:
            chunk = self.buffer[: self.window].reshape(1, -1)
            self.buffer = self.buffer[self.window :]
            x = np.concatenate((self.context, chunk), axis=1).astype(np.float32)
            out, state = self.session.run(
                None, {"input": x, "sr": self.sr, "state": self.state}
            )
            self.state = state
            self.context = chunk[:, -self.context_size :]
            if out[0][0] > self.threshold:
                detected = True
        return detected


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE, f"{path}: expected {SAMPLE_RATE} Hz"
        assert wf.getnchannels() == 1, f"{path}: expected mono"
        assert wf.getsampwidth() == 2, f"{path}: expected 16-bit"
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0


def segment_sentences(samples: np.ndarray, vad: VadSegmenter):
    """Yield sentence arrays using the same state machine as AudioWorker.run()."""
    vad.reset()
    holder = []
    pre_speech = collections.deque(maxlen=PADDING_CHUNKS)
    in_sentence = False
    silence_count = 0

    for start in range(0, len(samples) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
        chunk = samples[start : start + CHUNK_SAMPLES]
        if vad.is_speech(chunk):
            silence_count = 0
            if not in_sentence:
                in_sentence = True
                holder.extend(pre_speech)
                pre_speech.clear()
            holder.append(chunk)
        else:
            if in_sentence:
                silence_count += 1
                holder.append(chunk)
                if silence_count >= config_audio.SILENT_CHUNK_THRESHOLD:
                    vad.reset()
                    yield np.concatenate(holder)
                    holder = []
                    in_sentence = False
                    silence_count = 0
            else:
                pre_speech.append(chunk)

    if holder:  # trailing sentence at end of file
        yield np.concatenate(holder)


def transcribe_timed(model, sentence: np.ndarray):
    """Time exactly the calls AudioWorker makes per sentence."""
    t0 = time.perf_counter()
    ctx = model.transcribe_stream(
        context_size=(config_audio.CONTEXT_LEFT, config_audio.CONTEXT_RIGHT)
    )
    transcriber = ctx.__enter__()
    transcriber.add_audio(mx.array(sentence))
    text = transcriber.result.text.strip()
    ctx.__exit__(None, None, None)
    return time.perf_counter() - t0, text


def embed_timed(session, sentence: np.ndarray):
    t0 = time.perf_counter()
    emb = session.run(None, {"audio": sentence.reshape(1, -1)})[0][0]
    return time.perf_counter() - t0, emb


def main():
    wav_paths = sys.argv[1:] or DEFAULT_WAVS

    print(f"Loading Parakeet: {config_audio.PARAKEET_MODEL}")
    model = from_pretrained(config_audio.PARAKEET_MODEL)
    print(f"Loading ReDimNet: {config_audio.REDIMNET_PATH}")
    redimnet = ort.InferenceSession(config_audio.REDIMNET_PATH)
    vad = VadSegmenter()

    # Warmup: first inference includes weight load / kernel compile; exclude it.
    warmup = np.random.default_rng(0).normal(0, 0.01, SAMPLE_RATE * 2).astype(np.float32)
    t, _ = transcribe_timed(model, warmup)
    print(f"Warmup transcription (2.0 s audio, cold start): {t:.3f} s\n")
    embed_timed(redimnet, warmup)

    rows = []
    for path in wav_paths:
        samples = load_wav(path)
        for i, sentence in enumerate(segment_sentences(samples, vad)):
            dur = len(sentence) / SAMPLE_RATE
            stt_time, text = transcribe_timed(model, sentence)
            emb_time, _ = embed_timed(redimnet, sentence)
            rows.append((path, i, dur, stt_time, emb_time, text))

    name_w = max(len(p.split("/")[-1]) for p, *_ in rows) + 3
    print(f"{'file':<{name_w}} {'sent':>4} {'audio_s':>8} {'stt_s':>7} {'stt_RTF':>8} {'emb_s':>7} text")
    for path, i, dur, stt, emb, text in rows:
        snippet = (text[:48] + "...") if len(text) > 48 else text
        print(
            f"{path.split('/')[-1]:<{name_w}} {i:>4} {dur:>8.2f} {stt:>7.3f} "
            f"{stt / dur:>8.3f} {emb:>7.3f} {snippet!r}"
        )

    total_audio = sum(r[2] for r in rows)
    total_stt = sum(r[3] for r in rows)
    total_emb = sum(r[4] for r in rows)
    per_sentence_rtf = [r[3] / r[2] for r in rows]
    print(f"\nSentences: {len(rows)}   Total audio: {total_audio:.2f} s")
    print(f"Transcription: total {total_stt:.2f} s  ->  overall RTF {total_stt / total_audio:.3f}")
    print(f"  per-sentence RTF  mean {statistics.mean(per_sentence_rtf):.3f}  "
          f"median {statistics.median(per_sentence_rtf):.3f}  "
          f"min {min(per_sentence_rtf):.3f}  max {max(per_sentence_rtf):.3f}")
    print(f"Speaker embedding: total {total_emb:.2f} s  ->  overall RTF {total_emb / total_audio:.3f}")
    print(f"Combined STT+embedding overall RTF: {(total_stt + total_emb) / total_audio:.3f}")


if __name__ == "__main__":
    main()
