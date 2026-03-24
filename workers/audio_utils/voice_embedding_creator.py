import wave
import numpy as np
import onnxruntime as ort

def create_and_save_embedding(wav_file_path: str, onnx_model_path: str, output_npy_path: str):
    """
    Reads a mono 16kHz WAV file, generates a RedimNet speaker embedding, 
    and saves it to a .npy file.
    """
    
    # 1. Read the audio file
    with wave.open(wav_file_path, 'rb') as wf:
        # Note: This assumes the input is a 16kHz, mono, 16-bit PCM WAV file
        # to match the config_audio.AUDIO_SAMPLE_RATE_HZ in your system.
        raw_bytes = wf.readframes(wf.getnframes())
        
        # Convert int16 PCM to float32 (-1.0 to 1.0) exactly like your AudioWorker
        audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        
    # Reshape to a 2D array (1, num_samples) as required by the model
    audio_reshaped = audio_data.reshape(1, -1)

    # 2. Load the ONNX model (RedimNet)
    print("Loading RedimNet model...")
    session = ort.InferenceSession(onnx_model_path)

    # 3. Generate the embedding
    print("Generating embedding...")
    # This matches your self.get_embedding() method
    embedding = session.run(None, {"audio": audio_reshaped})[0][0]

    # 4. Save the embedding to a .npy file
    np.save(output_npy_path, embedding)
    print(f"Success! Embedding saved to: {output_npy_path}")


if __name__ == "__main__":
    # Replace these paths with your actual file paths
    WAV_INPUT = "workers/audio_utils/solo_voices/TigerGroupChat.wav"
    MODEL_PATH = "workers/audio_utils/redimnet_b2.onnx"
    NPY_OUTPUT = "database/sample_data/voice_Tiger2.npy"
    
    create_and_save_embedding(WAV_INPUT, MODEL_PATH, NPY_OUTPUT)