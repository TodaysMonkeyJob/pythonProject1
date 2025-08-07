import tempfile

from faster_whisper import WhisperModel, BatchedInferencePipeline
from typing import List, Union
import signal as system_signal
import torch.cuda as cuda
import soundfile as sf
import threading
from openai import OpenAI
import queue
import time
import os
import warnings
warnings.filterwarnings("ignore")
from system_setup.logger_setup import init_logger, set_logger_params
from .speech_config import TIME_SLEEP, INIT_MODEL_TRANSCRIPTION

logger = init_logger()
set_logger_params(logger, log_level="WARNING")


class TranscriptionWorker:
    def __init__(self,
                 conn,
                 stdout_pipe,
                 ready_event,
                 shutdown_event,
                 interrupt_stop_event,
                 api_connection,
                 model_path : str = INIT_MODEL_TRANSCRIPTION,
                 download_root: str = './model/',
                 compute_type: str = "default",
                 gpu_device_index: Union[int, List[int]] = 0,
                 device: str = "cuda",
                 beam_size: int = 5,
                 batch_size: int = 16,
                 faster_whisper_vad_filter: bool = True
                 ):
        print(api_connection)
        self.conn = conn
        self.stdout_pipe = stdout_pipe
        self.model_path = model_path if not api_connection else 'whisper-1'
        self.download_root = download_root or None
        self.compute_type = compute_type
        self.gpu_device_index = gpu_device_index
        self.device = "cuda" if device == "cuda" and cuda.is_available() else "cpu"
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.interrupt_stop_event = interrupt_stop_event
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.faster_whisper_vad_filter = faster_whisper_vad_filter
        self.queue = queue.Queue()
        self.api_connection = api_connection


    def api_client_connection(self):
        client = OpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        return client

    def custom_print(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        try:
            self.stdout_pipe.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass

    def poll_connection(self):
        while not self.shutdown_event.is_set():
            if self.conn.poll(0.01):
                try:
                    data = self.conn.recv()
                    self.queue.put(data)
                except Exception as e:
                    logger.error(f"Error receiving data from connection: {e}", exc_info=True)
            else:
                time.sleep(TIME_SLEEP)

    def run(self):
        if __name__ == "__main__":
            system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)
            __builtins__['print'] = self.custom_print

        logger.info(f"Initializing faster_whisper main transcription model {self.model_path}")

        if self.api_connection:
            model = self.api_client_connection()
            print(self.api_connection)
        else:
            try:
                model = WhisperModel(model_size_or_path=self.model_path, device=self.device,
                    compute_type=self.compute_type, device_index=self.gpu_device_index, download_root=self.download_root, )
                # Create a short dummy audio array, for example 1 second of silence at 16 kHz
                if self.batch_size > 0:
                    model = BatchedInferencePipeline(model=model)

                # Run a warm-up transcription
                current_dir = os.path.dirname(os.path.realpath(__file__))
                warmup_audio_path = os.path.join(current_dir, "warmup_audio.wav")
                warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
                segments, info = model.transcribe(warmup_audio_data, language="en", beam_size=1)
                model_warmup_transcription = " ".join(segment.text for segment in segments)
            except Exception as e:
                logger.exception(f"Error initializing main faster_whisper transcription model: {e}")
                raise

        self.ready_event.set()
        logger.debug("Faster_whisper main speech to text transcription model initialized successfully")

        # Start the polling thread
        polling_thread = threading.Thread(target=self.poll_connection)
        polling_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    audio, language = self.queue.get(timeout=0.1)
                    try:
                        logger.debug(f"Transcribing audio with language {language}")
                        if self.api_connection:
                            try:
                                # Створення тимчасового шляху до .wav файлу
                                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                                os.close(fd)  # Закриваємо дескриптор, бо soundfile сам відкриє файл

                                # Записуємо аудіо в тимчасовий файл
                                sf.write(tmp_path, audio, samplerate=16000)

                                # Відкриваємо файл для API
                                with open(tmp_path, "rb") as f:
                                    transcription = model.audio.transcriptions.create(
                                        model="whisper-1", file=f, language="en")

                            except Exception as e:
                                logger.error(f"Error in API-based transcription: {e}", exc_info=True)
                                self.conn.send(('error', str(e)))

                            finally:
                                try:
                                    os.remove(tmp_path)  # Видаляємо файл після завершення
                                except Exception as cleanup_err:
                                    logger.warning(f"Could not delete temp file: {cleanup_err}")
                        else:
                            if self.batch_size > 0:
                                segments, info = model.transcribe(audio, language=language if language else None,
                                    beam_size=self.beam_size, batch_size=self.batch_size,
                                    vad_filter=self.faster_whisper_vad_filter)
                            else:
                                segments, info = model.transcribe(audio, language=language if language else None,
                                    beam_size=self.beam_size, vad_filter=self.faster_whisper_vad_filter)
                            transcription = " ".join(seg.text for seg in segments).strip()
                        logger.debug(f"Final text detected with main model: {transcription}")
                        self.conn.send(('success', (transcription)))
                    except Exception as e:
                        logger.error(f"General error in transcription: {e}", exc_info=True)
                        self.conn.send(('error', str(e)))
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    self.interrupt_stop_event.set()
                    logger.debug("Transcription worker process finished due to KeyboardInterrupt")
                    break
                except Exception as e:
                    logger.error(f"General error in processing queue item: {e}", exc_info=True)
        finally:
            __builtins__['print'] = print  # Restore the original print function
            self.conn.close()
            self.stdout_pipe.close()
            self.shutdown_event.set()  # Ensure the polling thread will stop
            polling_thread.join()  # Wait for the polling thread to finish