from workers.base import BaseWorker
from parakeet_mlx import from_pretrained
import numpy as np
import mlx.core as mx
import time
import uuid
from core import config


class AudioWorker(BaseWorker):
    def setup(self):

        #so it doesn't read the config every single loop
        self.config = {
            "model": config.PARAKEET_MODEL,
            "chunk_ms": config.AUDIO_CHUNK_SIZE_MS,
            "sample_rate": config.AUDIO_SAMPLE_RATE_HZ,
            "context_left": config.CONTEXT_LEFT,
            "context_right": config.CONTEXT_RIGHT,
            "silent_chunks": config.SPEECH_CHUNK_SIZE,
            "loudness_threshold": config.LOUDNESS_THRESHOLD,
        }
        
        print(f"[AudioWorker] Loading model: {self.config['model']}")
        self.model = from_pretrained(self.config["model"])
        
        self.chunk_samples = int(self.config["sample_rate"] * self.config["chunk_ms"] / 1000)
        self.chunk_bytes = self.chunk_samples * 2
        
        print(f"[AudioWorker] Ready. Chunk: {self.config['chunk_ms']}ms ({self.chunk_bytes} bytes)")



    def speech_checker(self, speech) -> bool: 
        loudness = np.sqrt(np.mean(speech ** 2)) #since its a wave between 0 and 1 its squared so its positive, find mean and sqrt it to make it normalize
        return loudness > self.config["loudness_threshold"] #if the mean is very low, the user likely isnt talking


    def run(self):
        self.setup()
    
        audio_buffer = b""
        context_size = (self.config["context_left"], self.config["context_right"])
        
        #session state
        session_id = None
        transcriber = None
        ctx = None
        last_text = ""
        silence_count = 0  #track consecutive silent chunks
        
        try:
            while True:
                try:
                    raw_bytes = self.input_queue.get(timeout=1.0) # so it doesnt block forevers

                except:
                    continue
                audio_buffer += raw_bytes
                
                while len(audio_buffer) >= self.chunk_bytes:
                    chunk_bytes = audio_buffer[:self.chunk_bytes]
                    audio_buffer = audio_buffer[self.chunk_bytes:]
                    
                    samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                    is_speech = self.speech_checker(samples)  #check if speech
                    
                    if is_speech:
                        silence_count = 0  #reset silence counter cus speech
                        
                        # start new session if needed could start with no speech
                        if transcriber is None:
                            session_id = str(uuid.uuid4())[:8]
                            ctx = self.model.transcribe_stream(context_size=context_size)
                            transcriber = ctx.__enter__()
                            last_text = ""
                        
                        transcriber.add_audio(mx.array(samples))
                        text = transcriber.result.text.strip()

                        if text and text != last_text:
                            # self.output_queue.put({   #If we want to send it chunk by chunk not when it decides sentences break
                            #     "type": "speech",
                            #     "text": text,
                            #     "id": session_id,
                            #     "timestamp": time.time(),
                            #     "final": False
                            # })
                            last_text = text
                    
                    else:
                        #silence
                        if transcriber is not None:
                            silence_count += 1
                            
                            if silence_count >= self.config["silent_chunks"]:
                                #sentence break
                                if last_text:
                                    
                                    self.output_queue.put({
                                        "type": "speech",
                                        "text": last_text,
                                        "id": session_id,
                                        "timestamp": time.time(),
                                        "final": True
                                    })
                                #close session
                                ctx.__exit__(None, None, None)
                                transcriber = None
                                ctx = None
                                session_id = None
                                last_text = ""
                                silence_count = 0
        finally:
            if ctx:
                ctx.__exit__(None, None, None)


    def process(self, raw_bytes):
        # preprocess incoming data; dk form yet
        pass
