import torch.multiprocessing as mp
import signal as system_signal
import collections
import numpy as np
import traceback
import threading
import datetime
import base64
import queue
import halo
import time
import copy
import os
import re
import gc
import psutil
from system_setup.logger_setup import init_logger, set_logger_params
from .transcription_worker import TranscriptionWorker
from .recognize_speaker import SpeakerRecognizer
from .vad_silero import SileroVAD
from .vad_webrtc import WebRTCVAD
from .speech_config import *
import warnings
warnings.filterwarnings("ignore")
process = psutil.Process(os.getpid())
before = process.memory_info().rss

logger = init_logger()
# Set OpenMP runtime duplicate library handling to OK (Use only for development!)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class AudioToTextRecorder:
    """
    A class responsible for capturing audio from the microphone, detecting
    voice activity, and then transcribing the captured audio using the
    `faster_whisper` model.
    """

    def __init__(self,
                 language: str = "en",
                 input_device_index: int = None,
                 on_recording_start=None,
                 on_recording_stop=None,
                 on_transcription_start=None,
                 spinner=True,
                 log_level="WARNING",
                 api_connection=False,

                 # Voice activation parameters
                 silero_deactivity_detection: bool = False,
                 post_speech_silence_duration: float = (INIT_POST_SPEECH_SILENCE_DURATION),
                 min_length_of_recording: float = (INIT_MIN_LENGTH_OF_RECORDING),
                 min_gap_between_recordings: float = (INIT_MIN_GAP_BETWEEN_RECORDINGS),
                 on_vad_start=None, on_vad_detect_start=None,
                 on_vad_stop=None, on_vad_detect_stop=None,

                 on_recorded_chunk=None,
                 handle_buffer_overflow: bool = INIT_HANDLE_BUFFER_OVERFLOW,
                 buffer_size: int = BUFFER_SIZE, sample_rate: int = SAMPLE_RATE,
                 allowed_latency_limit: int = ALLOWED_LATENCY_LIMIT):


        # === Audio Configuration ===
        self.audio_interface = None
        self.audio = None
        self.stream = None
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_queue = mp.Queue()
        self.last_words_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) * 0.3))

        # === Recording Control Flags ===
        self.is_recording = False
        self.is_running = True
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False

        # === Voice Activity Detection (VAD) ===
        self.silero_deactivity_detection = silero_deactivity_detection
        self.min_length_of_recording = min_length_of_recording
        self.post_speech_silence_duration = post_speech_silence_duration
        self.frames = []
        self.last_frames = []
        self.audio_buffer = collections.deque(
            maxlen=int(self.sample_rate // self.buffer_size * INIT_POST_SPEECH_SILENCE_DURATION))
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self.speech_end_silence_start = 0
        self.silero_working = False
        self.on_vad_start = on_vad_start
        self.on_vad_stop = on_vad_stop
        self.on_vad_detect_start = on_vad_detect_start
        self.on_vad_detect_stop = on_vad_detect_stop

        # === Transcription ===
        self.language = language
        self.on_transcription_start = on_transcription_start
        self.parent_transcription_pipe, self.child_transcription_pipe = mp.Pipe()
        self.parent_stdout_pipe, self.child_stdout_pipe = mp.Pipe()
        self.transcription_lock = threading.Lock()
        self.transcribe_count = 0
        self.early_transcription_on_silence = 0
        self.last_transcription_bytes = None
        self.last_transcription_bytes_b64 = None
        self.api_connection = api_connection

        # === Thread and Process Management ===
        self.recording_thread = None
        self.stdout_thread = None
        self.shutdown_event = mp.Event()
        self.shutdown_lock = threading.Lock()
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.interrupt_stop_event = mp.Event()
        self.was_interrupted = mp.Event()
        self.main_transcription_ready_event = mp.Event()

        # === State and Control ===
        self.state = "inactive"
        self.is_shut_down = False
        self.spinner = spinner
        self.halo = None
        self.input_device_index = input_device_index
        self.handle_buffer_overflow = handle_buffer_overflow
        self.allowed_latency_limit = allowed_latency_limit
        self.min_gap_between_recordings = min_gap_between_recordings
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.last_recording_start_time = 0
        self.last_recording_stop_time = 0
        self.backdate_stop_seconds = 0.0
        self.backdate_resume_seconds = 0.0
        self.silero_check_time = 0
        self.listen_start = 0

        # === Callbacks ===
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_recorded_chunk = on_recorded_chunk

        # === Logging ===
        self.log_level = log_level
        set_logger_params(logger, log_file='IoTsecuritySTTx.log', log_level="WARNING", no_log_file=False)
        logger.info("Starting IoT security voice assistant")


        self._initialize_multiprocessing()
        self.initialize_vad()
        self.initialize_speaker_recognizer()
        self.initialize_workers()
        self._wait_for_transcription_model()
        self._initialize_threads()

        logger.debug('IoT security initialization completed successfully')

    def _initialize_multiprocessing(self):
        """Initialize multiprocessing safely."""
        try:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn")
        except RuntimeError as e:
            logger.info(f"Multiprocessing start method already set. Details: {e}")

    def initialize_vad(self):
        """Initialize WebRTC and Silero VAD models."""
        # Setup voice activity detection models
        try:
            self.webrtc_vad_model = WebRTCVAD(state_callback=self._update_webrtc_state)
            logger.debug("WebRTC VAD voice activity detection engine initialized successfully")
            self.silero_vad_model = SileroVAD(state_callback=self._update_silero_state)
            logger.debug("Silero VAD voice activity detection engine initialized successfully")
        except Exception as e:
            logger.exception("Error initializing WebRTC or Silero voice activity detection engine: {e}")
            raise

    def initialize_speaker_recognizer(self):
        """Initialize SpeakerRecognizer models."""
        try:
            self.speaker_recognizer = SpeakerRecognizer()
            logger.debug("SpeakerRecognizer engine initialized successfully")
        except Exception as e:
            logger.exception("Error initializing SpeakerRecognizer engine: {e}")
            raise

    def initialize_workers(self):
        """
        Initialize and start all necessary workers (processes and threads).
        """
        # Start transcription worker as process
        self.transcript_process = self._start_thread(
            target=AudioToTextRecorder._transcription_worker,
            args=(
                self.child_transcription_pipe,
                self.child_stdout_pipe,
                self.main_transcription_ready_event,
                self.shutdown_event,
                self.interrupt_stop_event,
                self.api_connection)
        )

        # Start audio data worker as process
        self.reader_process = self._start_thread(
            target=AudioToTextRecorder._audio_data_worker,
            args=(
                self.audio_queue,
                self.sample_rate,
                self.buffer_size,
                self.input_device_index,
                self.shutdown_event,
                self.interrupt_stop_event)
        )

    def _wait_for_transcription_model(self):
        """Wait for transcription model to start."""
        logger.debug("Waiting for transcription model to start...")
        self.main_transcription_ready_event.wait()
        logger.debug("Transcription model initialization completed.")

    def _initialize_threads(self):
        """
        Initializes and starts a threads.
        """
        # Start the recording worker thread
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        # Start the stdout worker thread
        self.stdout_thread = threading.Thread(target=self._read_stdout)
        self.stdout_thread.daemon = True
        self.stdout_thread.start()

    def _start_thread(self, target=None, args=()):
        """
        Implement a consistent threading model across the library.

        This method is used to start any thread in this library. It uses the
        standard threading. Thread for Linux and for all others uses the pytorch
        MultiProcessing library 'Process'.
        Args:
            target (callable object): is the callable object to be invoked by
              the run() method. Defaults to None, meaning nothing is called.
            args (tuple): is a list or tuple of arguments for the target
              invocation. Defaults to ().
        """
        if (platform.system() == 'Linux'):
            thread = threading.Thread(target=target, args=args)
            thread.deamon = True
            thread.start()
            return thread
        else:
            thread = mp.Process(target=target, args=args)
            thread.start()
            return thread

    def _read_stdout(self):
        while not self.shutdown_event.is_set():
            try:
                if self.parent_stdout_pipe.poll(0.1):
                    logger.debug("Receive from stdout pipe")
                    message = self.parent_stdout_pipe.recv()
                    logger.info(message)
            except (BrokenPipeError, EOFError, OSError):
                # The pipe probably has been closed, so we ignore the error
                pass
            except KeyboardInterrupt:  # handle manual interruption (Ctrl+C)
                logger.info("KeyboardInterrupt in read from stdout detected, exiting...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in read from stdout: {e}", exc_info=True)
                logger.error(traceback.format_exc())  # Log the full traceback here
                break
            time.sleep(0.1)

    def _transcription_worker(*args, **kwargs):
        worker = TranscriptionWorker(*args, **kwargs)
        worker.run()

    @staticmethod
    def _audio_data_worker(audio_queue, target_sample_rate, buffer_size, input_device_index, shutdown_event,
            interrupt_stop_event):
        """
        Worker method that handles the audio recording process.

        This method runs in a separate process and is responsible for:
        - Setting up the audio input stream for recording at the highest possible sample rate.
        - Continuously reading audio data from the input stream, resampling if necessary,
        preprocessing the data, and placing complete chunks in a queue.
        - Handling errors during the recording process.
        - Gracefully terminating the recording process when a shutdown event is set.

        Args:
            audio_queue (queue.Queue): A queue where recorded audio data is placed.
            target_sample_rate (int): The desired sample rate for the output audio (for Silero VAD).
            buffer_size (int): The number of samples expected by the Silero VAD model.
            input_device_index (int): The index of the audio input device.
            shutdown_event (threading.Event): An event that, when set, signals this worker method to terminate.
            interrupt_stop_event (threading.Event): An event to signal keyboard interrupt.

        Raises:
            Exception: If there is an error while initializing the audio recording.
        """
        import pyaudio
        import numpy as np
        from scipy import signal

        if __name__ == '__main__':
            system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)

        def get_highest_sample_rate(audio_interface, device_index):
            """Get the highest supported sample rate for the specified device."""
            try:
                device_info = audio_interface.get_device_info_by_index(device_index)
                logger.debug(f"Retrieving highest sample rate for device index {device_index}: {device_info}")
                max_rate = int(device_info['defaultSampleRate'])

                if 'supportedSampleRates' in device_info:
                    supported_rates = [int(rate) for rate in device_info['supportedSampleRates']]
                    if supported_rates:
                        max_rate = max(supported_rates)

                logger.debug(f"Highest supported sample rate for device index {device_index} is {max_rate}")
                return max_rate
            except Exception as e:
                logger.warning(f"Failed to get highest sample rate: {e}")
                return 48000  # Fallback to a common high sample rate

        def initialize_audio_stream(audio_interface, sample_rate, chunk_size):
            nonlocal input_device_index

            def validate_device(device_index):
                """Validate that the device exists and is actually available for input."""
                try:
                    device_info = audio_interface.get_device_info_by_index(device_index)
                    logger.debug(f"Validating device index {device_index} with info: {device_info}")
                    if not device_info.get('maxInputChannels', 0) > 0:
                        logger.debug("Device has no input channels, invalid for recording.")
                        return False

                    # Try to actually read from the device
                    test_stream = audio_interface.open(format=pyaudio.paInt16, channels=1, rate=target_sample_rate,
                        input=True, frames_per_buffer=chunk_size, input_device_index=device_index, start=False
                        # Don't start the stream yet
                    )

                    test_stream.start_stream()
                    test_data = test_stream.read(chunk_size, exception_on_overflow=False)
                    test_stream.stop_stream()
                    test_stream.close()

                    if len(test_data) == 0:
                        logger.debug("Device produced no data, invalid for recording.")
                        return False

                    logger.debug(f"Device index {device_index} successfully validated.")
                    return True

                except Exception as e:
                    logger.debug(f"Device validation failed for index {device_index}: {e}")
                    return False

            """Initialize the audio stream with error handling."""
            while not shutdown_event.is_set():
                try:
                    # First, get a list of all available input devices
                    input_devices = []
                    device_count = audio_interface.get_device_count()
                    logger.debug(f"Found {device_count} total audio devices on the system.")
                    for i in range(device_count):
                        try:
                            device_info = audio_interface.get_device_info_by_index(i)
                            if device_info.get('maxInputChannels', 0) > 0:
                                input_devices.append(i)
                        except Exception as e:
                            logger.debug(f"Could not retrieve info for device index {i}: {e}")
                            continue

                    logger.debug(f"Available input devices with input channels: {input_devices}")
                    if not input_devices:
                        raise Exception("No input devices found")

                    # If input_device_index is None or invalid, try to find a working device
                    if input_device_index is None or input_device_index not in input_devices:
                        # First try the default device
                        try:
                            default_device = audio_interface.get_default_input_device_info()
                            logger.debug(f"Default device info: {default_device}")
                            if validate_device(default_device['index']):
                                input_device_index = default_device['index']
                                logger.debug(f"Default device {input_device_index} selected.")
                        except Exception:
                            # If default device fails, try other available input devices
                            logger.debug("Default device validation failed, checking other devices...")
                            for device_index in input_devices:
                                if validate_device(device_index):
                                    input_device_index = device_index
                                    logger.debug(f"Device {input_device_index} selected.")
                                    break
                            else:
                                raise Exception("No working input devices found")

                    # Validate the selected device one final time
                    if not validate_device(input_device_index):
                        raise Exception("Selected device validation failed")

                    # If we get here, we have a validated device
                    logger.debug(f"Opening stream with device index {input_device_index}, "
                                 f"sample_rate={sample_rate}, chunk_size={chunk_size}")
                    stream = audio_interface.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True,
                        frames_per_buffer=chunk_size, input_device_index=input_device_index, )

                    logger.info(f"Microphone connected and validated (device index: {input_device_index}, "
                                f"sample rate: {sample_rate}, chunk size: {chunk_size})")
                    return stream

                except Exception as e:
                    logger.error(f"Microphone connection failed: {e}. Retrying...", exc_info=True)
                    input_device_index = None
                    time.sleep(3)  # Wait before retrying
                    continue

        def preprocess_audio(chunk, original_sample_rate, target_sample_rate):
            """Preprocess audio chunk similar to feed_audio method."""
            if isinstance(chunk, np.ndarray):
                # Handle stereo to mono conversion if necessary
                if chunk.ndim == 2:
                    chunk = np.mean(chunk, axis=1)

                # Resample to target_sample_rate if necessary
                if original_sample_rate != target_sample_rate:
                    logger.debug(f"Resampling from {original_sample_rate} Hz to {target_sample_rate} Hz.")
                    num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                    chunk = signal.resample(chunk, num_samples)

                chunk = chunk.astype(np.int16)
            else:
                # If chunk is bytes, convert to numpy array
                chunk = np.frombuffer(chunk, dtype=np.int16)

                # Resample if necessary
                if original_sample_rate != target_sample_rate:
                    logger.debug(f"Resampling from {original_sample_rate} Hz to {target_sample_rate} Hz.")
                    num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                    chunk = signal.resample(chunk, num_samples)
                    chunk = chunk.astype(np.int16)

            return chunk.tobytes()

        audio_interface = None
        stream = None
        device_sample_rate = None
        chunk_size = 1024  # Increased chunk size for better performance

        def setup_audio():
            nonlocal audio_interface, stream, device_sample_rate, input_device_index
            try:
                if audio_interface is None:
                    logger.debug("Creating PyAudio interface...")
                    audio_interface = pyaudio.PyAudio()

                if input_device_index is None:
                    try:
                        default_device = audio_interface.get_default_input_device_info()
                        input_device_index = default_device['index']
                        logger.debug(f"No device index supplied; using default device {input_device_index}")
                    except OSError as e:
                        logger.debug(f"Default device retrieval failed: {e}")
                        input_device_index = None

                # We'll try 16000 Hz first, then the highest rate we detect, then fallback if needed
                sample_rates_to_try = [16000]
                if input_device_index is not None:
                    highest_rate = get_highest_sample_rate(audio_interface, input_device_index)
                    if highest_rate != 16000:
                        sample_rates_to_try.append(highest_rate)
                else:
                    sample_rates_to_try.append(48000)

                logger.debug(f"Sample rates to try for device {input_device_index}: {sample_rates_to_try}")

                for rate in sample_rates_to_try:
                    try:
                        device_sample_rate = rate
                        logger.debug(f"Attempting to initialize audio stream at {device_sample_rate} Hz.")
                        stream = initialize_audio_stream(audio_interface, device_sample_rate, chunk_size)
                        if stream is not None:
                            logger.debug(f"Audio recording initialized successfully at {device_sample_rate} Hz, "
                                         f"reading {chunk_size} frames at a time")
                            return True
                    except Exception as e:
                        logger.warning(f"Failed to initialize audio stream at {device_sample_rate} Hz: {e}")
                        continue

                # If we reach here, none of the sample rates worked
                raise Exception("Failed to initialize audio stream with all sample rates.")

            except Exception as e:
                logger.exception(f"Error initializing pyaudio audio recording: {e}")
                if audio_interface:
                    audio_interface.terminate()
                return False

        logger.debug(f"Starting audio data worker with target_sample_rate={target_sample_rate}, "
                     f"buffer_size={buffer_size}, input_device_index={input_device_index}")

        if not setup_audio():
            raise Exception("Failed to set up audio recording.")

        buffer = bytearray()
        silero_buffer_size = 2 * buffer_size  # Silero complains if too short

        time_since_last_buffer_message = 0

        try:
            while not shutdown_event.is_set():
                try:
                    data = stream.read(chunk_size, exception_on_overflow=False)

                    processed_data = preprocess_audio(data, device_sample_rate, target_sample_rate)
                    buffer += processed_data

                    # Check if the buffer has reached or exceeded the silero_buffer_size
                    while len(buffer) >= silero_buffer_size:
                        # Extract silero_buffer_size amount of data from the buffer
                        to_process = buffer[:silero_buffer_size]
                        buffer = buffer[silero_buffer_size:]

                        # Feed the extracted data to the audio_queue
                        if time_since_last_buffer_message:
                            time_passed = time.time() - time_since_last_buffer_message
                            if time_passed > 1:
                                logger.debug("_audio_data_worker writing audio data into queue.")
                                time_since_last_buffer_message = time.time()
                        else:
                            time_since_last_buffer_message = time.time()

                        audio_queue.put(to_process)

                except OSError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        logger.warning("Input overflowed. Frame dropped.")
                    else:
                        logger.error(f"OSError during recording: {e}", exc_info=True)
                        # Attempt to reinitialize the stream
                        logger.error("Attempting to reinitialize the audio stream...")

                        try:
                            if stream:
                                stream.stop_stream()
                                stream.close()
                        except Exception:
                            pass

                        time.sleep(1)
                        if not setup_audio():
                            logger.error("Failed to reinitialize audio stream. Exiting.")
                            break
                        else:
                            logger.error("Audio stream reinitialized successfully.")
                    continue

                except Exception as e:
                    logger.error(f"Unknown error during recording: {e}")
                    tb_str = traceback.format_exc()
                    logger.error(f"Traceback: {tb_str}")
                    logger.error(f"Error: {e}")
                    # Attempt to reinitialize the stream
                    logger.info("Attempting to reinitialize the audio stream...")
                    try:
                        if stream:
                            stream.stop_stream()
                            stream.close()
                    except Exception:
                        pass

                    time.sleep(1)
                    if not setup_audio():
                        logger.error("Failed to reinitialize audio stream. Exiting.")
                        break
                    else:
                        logger.info("Audio stream reinitialized successfully.")
                    continue

        except KeyboardInterrupt:
            interrupt_stop_event.set()
            logger.debug("Audio data worker process finished due to KeyboardInterrupt")
        finally:
            # After recording stops, feed any remaining audio data
            if buffer:
                audio_queue.put(bytes(buffer))

            try:
                if stream:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            if audio_interface:
                audio_interface.terminate()

    def abort(self):
        state = self.state
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self.interrupt_stop_event.set()
        if self.state != "inactive":  # if inactive, was_interrupted will never be set
            self.was_interrupted.wait()
            self._set_state("transcribing")
        self.was_interrupted.clear()
        if self.is_recording:  # if recording, make sure to stop the recorder
            self.stop()

    def wait_audio(self):
        """
        Waits for the start and completion of the audio recording process.

        This method is responsible for:
        - Waiting for voice activity to begin recording if not yet started.
        - Waiting for voice inactivity to complete the recording.
        - Setting the audio buffer from the recorded frames.
        - Resetting recording-related attributes.

        Side effects:
        - Updates the state of the instance.
        - Modifies the audio attribute to contain the processed audio data.
        """

        try:
            logger.info("Setting listen time")
            if self.listen_start == 0:
                self.listen_start = time.time()

            # If not yet started recording, wait for voice activity to initiate.
            if not self.is_recording and not self.frames:
                self._set_state("listening")
                self.start_recording_on_voice_activity = True

                # Wait until recording starts
                logger.debug('Waiting for recording start')
                while not self.interrupt_stop_event.is_set():
                    if self.start_recording_event.wait(timeout=0.02):
                        break

            # If recording is ongoing, wait for voice inactivity
            # to finish recording.
            if self.is_recording:
                self.stop_recording_on_voice_deactivity = True

                # Wait until recording stops
                logger.debug('Waiting for recording stop')
                while not self.interrupt_stop_event.is_set():
                    if (self.stop_recording_event.wait(timeout=0.02)):
                        break

            frames = self.frames
            if len(frames) == 0:
                frames = self.last_frames

            # Calculate samples needed for backdating resume
            samples_to_keep = int(self.sample_rate * self.backdate_resume_seconds)

            # First convert all current frames to audio array
            full_audio_array = np.frombuffer(b''.join(frames), dtype=np.int16)
            full_audio = full_audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE

            # Calculate how many samples we need to keep for backdating resume
            if samples_to_keep > 0:
                samples_to_keep = min(samples_to_keep, len(full_audio))
                # Keep the last N samples for backdating resume
                frames_to_read_audio = full_audio[-samples_to_keep:]

                # Convert the audio back to int16 bytes for frames
                frames_to_read_int16 = (frames_to_read_audio * INT16_MAX_ABS_VALUE).astype(np.int16)
                frame_bytes = frames_to_read_int16.tobytes()

                # Split into appropriate frame sizes (assuming standard frame size)
                FRAME_SIZE = 2048  # Typical frame size
                frames_to_read = []
                for i in range(0, len(frame_bytes), FRAME_SIZE):
                    frame = frame_bytes[i:i + FRAME_SIZE]
                    if frame:  # Only add non-empty frames
                        frames_to_read.append(frame)
            else:
                frames_to_read = []

            # Process backdate stop seconds
            samples_to_remove = int(self.sample_rate * self.backdate_stop_seconds)

            if samples_to_remove > 0:
                if samples_to_remove < len(full_audio):
                    self.audio = full_audio[:-samples_to_remove]
                    logger.debug(f"Removed {samples_to_remove} samples "
                                 f"({samples_to_remove / self.sample_rate:.3f}s) from end of audio")
                else:
                    self.audio = np.array([], dtype=np.float32)
                    logger.debug("Cleared audio (samples_to_remove >= audio length)")
            else:
                self.audio = full_audio
                logger.debug(f"No samples removed, final audio length: {len(self.audio)}")
            self.frames.clear()
            self.last_frames.clear()
            self.frames.extend(frames_to_read)

            # Reset backdating parameters
            self.backdate_stop_seconds = 0.0
            self.backdate_resume_seconds = 0.0

            self.listen_start = 0

            self._set_state("inactive")

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt in wait_audio, shutting down")
            self.shutdown()
            raise  # Re-raise the exception after cleanup

    def transcribe(self):
        """
        Transcribes audio captured by this class instance using the
        `faster_whisper` model.

        Automatically starts recording upon voice activity if not manually
          started using `recorder.start()`.
        Automatically stops recording upon voice deactivity if not manually
          stopped with `recorder.stop()`.
        Processes the recorded audio to generate transcription.

        Args:
            on_transcription_finished (callable, optional): Callback function
              to be executed when transcription is ready.
            If provided, transcription will be performed asynchronously,
              and the callback will receive the transcription as its argument.
              If omitted, the transcription will be performed synchronously,
              and the result will be returned.

        Returns (if no callback is set):
            str: The transcription of the recorded audio.

        Raises:
            Exception: If there is an error during the transcription process.
        """

        self._set_state("transcribing")
        audio_copy = copy.deepcopy(self.audio)
        main_speaker = self.speaker_recognizer.identify_speaker(audio_copy)
        start_time = 0
        with self.transcription_lock:

            try:
                if self.transcribe_count == 0:
                    logger.debug("Adding transcription request, no early transcription started")
                    start_time = time.time()  # Start timing
                    self.parent_transcription_pipe.send((audio_copy, self.language))
                    self.transcribe_count += 1

                while self.transcribe_count > 0:
                    if not self.parent_transcription_pipe.poll(0.1):  # check if transcription done
                        if self.interrupt_stop_event.is_set():  # check if interrupted
                            self.was_interrupted.set()
                            self._set_state("inactive")
                            return ""  # return empty string if interrupted
                        continue
                    status, result = self.parent_transcription_pipe.recv()
                    self.transcribe_count -= 1

                self.allowed_to_early_transcribe = True
                self._set_state("inactive")
                if status == 'success':
                    segments  = result
                    self.last_transcription_bytes = copy.deepcopy(audio_copy)
                    self.last_transcription_bytes_b64 = base64.b64encode(
                    self.last_transcription_bytes.tobytes()).decode('utf-8')
                    if self.api_connection:
                        transcription = segments.text
                    else:
                        transcription = self._preprocess_output(segments)
                    end_time = time.time()  # End timing
                    transcription_time = end_time - start_time

                    if start_time:
                        logger.debug(
                            f"Model completed transcription in {transcription_time:.2f} seconds")
                        logger.info(f"Text after transcription: {transcription}; {main_speaker}")
                        after = process.memory_info().rss

                        used_memory_mb = (after - before) / 1024 / 1024
                        logger.info(f"Memory used for transcription: {used_memory_mb:.2f} MB")
                    return "" if self.interrupt_stop_event.is_set() else f"{transcription}; {main_speaker}"  # if interrupted return empty string
                else:
                    logger.error(f"Transcription error: {result}")
                    raise Exception(result)
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}", exc_info=True)
                raise e


    def text(self, on_transcription_finished=None, ):
        """
        Transcribes audio captured by this class instance
        using the `faster_whisper` model.

        - Automatically starts recording upon voice activity if not manually
          started using `recorder.start()`.
        - Automatically stops recording upon voice deactivity if not manually
          stopped with `recorder.stop()`.
        - Processes the recorded audio to generate transcription.

        Args:
            on_transcription_finished (callable, optional): Callback function
              to be executed when transcription is ready.
            If provided, transcription will be performed asynchronously, and
              the callback will receive the transcription as its argument.
              If omitted, the transcription will be performed synchronously,
              and the result will be returned.

        Returns (if not callback is set):
            str: The transcription of the recorded audio
        """
        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()
        try:
            self.wait_audio()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt in text() method")
            self.shutdown()
            raise  # Re-raise the exception after cleanup

        if self.is_shut_down or self.interrupt_stop_event.is_set():
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
            return ""

        if on_transcription_finished:
            threading.Thread(target=on_transcription_finished, args=(self.transcribe(),)).start()
        else:
            return self.transcribe()

    def start(self, frames=None):
        """
        Starts recording audio directly without waiting for voice activity.
        """

        # Ensure there's a minimum interval
        # between stopping and starting recording
        if (time.time() - self.recording_stop_time < self.min_gap_between_recordings):
            logger.info("Attempted to start recording "
                        "too soon after stopping.")
            return self

        logger.info("recording started")
        self._set_state("recording")
        self.frames = []
        if frames:
            self.frames = frames
        self.is_recording = True

        self.recording_start_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.stop_recording_event.clear()
        self.start_recording_event.set()

        if self.on_recording_start:
            self.on_recording_start()

        return self

    def stop(self, backdate_stop_seconds: float = 0.0, backdate_resume_seconds: float = 0.0, ):
        """
        Stops recording audio.

        Args:
        - backdate_stop_seconds (float, default="0.0"): Specifies the number of
            seconds to backdate the stop time. This is useful when the stop
            command is issued after the actual stop time.
        - backdate_resume_seconds (float, default="0.0"): Specifies the number
            of seconds to backdate the time relistening is initiated.
        """

        # Ensure there's a minimum interval
        # between starting and stopping recording
        if (time.time() - self.recording_start_time < self.min_length_of_recording):
            logger.info("Attempted to stop recording "
                        "too soon after starting.")
            return self

        logger.info("recording stopped")
        self.last_frames = copy.deepcopy(self.frames)
        self.backdate_stop_seconds = backdate_stop_seconds
        self.backdate_resume_seconds = backdate_resume_seconds
        self.is_recording = False
        self.recording_stop_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.silero_check_time = 0
        self.start_recording_event.clear()
        self.stop_recording_event.set()

        self.last_recording_start_time = self.recording_start_time
        self.last_recording_stop_time = self.recording_stop_time

        if self.on_recording_stop:
            self.on_recording_stop()

        return self

    def shutdown(self):
        """
        Safely shuts down the audio recording by stopping the
        recording worker and closing the audio stream.
        """

        with self.shutdown_lock:
            if self.is_shut_down:
                return

            print("\033[91mIoT security shutting down\033[0m")

            # Force wait_audio() and text() to exit
            self.is_shut_down = True
            self.start_recording_event.set()
            self.stop_recording_event.set()

            self.shutdown_event.set()
            self.is_recording = False
            self.is_running = False

            logger.debug('Finishing recording thread')
            if self.recording_thread:
                self.recording_thread.join()

            logger.debug('Terminating reader process')

            # Give it some time to finish the loop and cleanup.
            self.reader_process.join(timeout=10)

            if self.reader_process.is_alive():
                logger.warning("Reader process did not terminate "
                               "in time. Terminating forcefully.")
                self.reader_process.terminate()

            logger.debug('Terminating transcription process')
            self.transcript_process.join(timeout=10)

            if self.transcript_process.is_alive():
                logger.warning("Transcript process did not terminate "
                               "in time. Terminating forcefully.")
                self.transcript_process.terminate()

            self.parent_transcription_pipe.close()

            gc.collect()

    def _recording_worker(self):
        """
        The main worker method which constantly monitors the audio
        input for voice activity and accordingly starts/stops the recording.
        """

        last_inner_try_time = 0
        try:
            time_since_last_buffer_message = 0
            was_recording = False
            self.allowed_to_early_transcribe = True

            # Continuously monitor audio for voice activity
            while self.is_running:

                if last_inner_try_time:
                    last_processing_time = time.time() - last_inner_try_time
                    if last_processing_time > 0.1:
                        logger.warning('### WARNING: PROCESSING TOOK TOO LONG')
                last_inner_try_time = time.time()
                try:
                    try:
                        data = self.audio_queue.get(timeout=0.01)
                        self.last_words_buffer.append(data)
                    except queue.Empty:
                        if not self.is_running:
                            break
                        continue

                    if self.on_recorded_chunk:
                        self.on_recorded_chunk(data)

                    if self.handle_buffer_overflow:
                        current_size = self._audio_queue_size()
                        # Handle queue overflow
                        if current_size is not None and current_size > self.allowed_latency_limit:
                            logger.warning("Audio queue size exceeds "
                                           "latency limit. Current size: "
                                           f"{current_size}. "
                                           "Discarding old audio chunks.")

                            while current_size > self.allowed_latency_limit:
                                try:
                                    self.audio_queue.get_nowait()
                                except queue.Empty:
                                    break
                                current_size -= 1

                except BrokenPipeError:
                    logger.error("BrokenPipeError _recording_worker", exc_info=True)
                    self.is_running = False
                    break

                # Feed the extracted data to the audio_queue
                if time_since_last_buffer_message:
                    time_passed = time.time() - time_since_last_buffer_message
                    if time_passed > 1:
                        time_since_last_buffer_message = time.time()
                else:
                    time_since_last_buffer_message = time.time()

                failed_stop_attempt = False

                if not self.is_recording:

                    # Set state and spinner text
                    if not self.recording_stop_time:
                        if self.listen_start:
                            self._set_state("listening")
                        else:
                            self._set_state("inactive")

                    # Check for voice activity to
                    # trigger the start of recording
                    if self.start_recording_on_voice_activity:

                        if self._is_voice_active():
                            if self.on_vad_start:
                                self.on_vad_start()

                            logger.info("voice activity detected")
                            self.start()
                            self.start_recording_on_voice_activity = False
                            # Add the buffered audio
                            # to the recording frames
                            self.frames.extend(list(self.audio_buffer))
                            self.audio_buffer.clear()
                            self.silero_vad_model.reset_states()
                        else:
                            data_copy = data[:]
                            self._check_voice_activity(data_copy)
                    self.speech_end_silence_start = 0

                else:
                    # Stop the recording if silence is detected after speech
                    if self.stop_recording_on_voice_deactivity:
                        is_speech = (self.silero_vad_model.is_voice_active(
                            data) if self.silero_deactivity_detection else self.webrtc_vad_model.is_voice_active(data, True))

                        if not is_speech:
                            # Voice deactivity was detected, so we start
                            # measuring silence time before stopping recording
                            if self.speech_end_silence_start == 0 and (
                                    time.time() - self.recording_start_time > self.min_length_of_recording):
                                self.speech_end_silence_start = time.time()

                            if self.speech_end_silence_start and self.early_transcription_on_silence and len(
                                    self.frames) > 0 and (
                                    time.time() - self.speech_end_silence_start > self.early_transcription_on_silence) and self.allowed_to_early_transcribe:
                                self.transcribe_count += 1
                                audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
                                audio = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
                                self.parent_transcription_pipe.send((audio, self.language))
                                self.allowed_to_early_transcribe = False

                        else:
                            if self.speech_end_silence_start:
                                self.speech_end_silence_start = 0
                                self.allowed_to_early_transcribe = True

                        # Wait for silence to stop recording after speech
                        if self.speech_end_silence_start and time.time() - self.speech_end_silence_start >= self.post_speech_silence_duration:
                            if self.on_vad_stop:
                                self.on_vad_stop()

                            # Get time in desired format (HH:MM:SS.nnn)
                            silence_start_time = datetime.datetime.fromtimestamp(
                                self.speech_end_silence_start).strftime('%H:%M:%S.%f')[:-3]

                            # Calculate time difference
                            time_diff = time.time() - self.speech_end_silence_start

                            logger.info(f"voice deactivity detected at {silence_start_time}, "
                                        f"time since silence start: {time_diff:.3f} seconds")
                            self.frames.append(data)
                            self.stop()
                            if not self.is_recording:
                                self.speech_end_silence_start = 0
                            else:
                                failed_stop_attempt = True

                if not self.is_recording and was_recording:
                    # Reset after stopping recording to ensure clean state
                    self.stop_recording_on_voice_deactivity = False

                if time.time() - self.silero_check_time > 0.1:
                    self.silero_check_time = 0
                was_recording = self.is_recording

                if self.is_recording and not failed_stop_attempt:
                    self.frames.append(data)

                if not self.is_recording or self.speech_end_silence_start:
                    self.audio_buffer.append(data)

        except Exception as e:
            logger.debug('Debug: Caught exception in main try block')
            if not self.interrupt_stop_event.is_set():
                logger.error(f"Unhandled exeption in _recording_worker: {e}", exc_info=True)
                raise

    def _update_silero_state(self, working: bool, is_speech_active: bool):
        """
        Update the working state of Silero VAD.
        """
        self.silero_working = working
        if is_speech_active is not None:
            self.is_silero_speech_active = is_speech_active

    def _update_webrtc_state(self, is_speech_active: bool):
        """
        Update the working state of WebRTC VAD.
        """
        if is_speech_active is not None:
            self.is_webrtc_speech_active = is_speech_active

    def _check_voice_activity(self, data):
        """
        Initiate check if voice is active based on the provided data.

        Args:
            data: The audio data to be checked for voice activity.
        """
        self.webrtc_vad_model.is_voice_active(data)

        # First quick performing check for voice activity using WebRTC
        if self.is_webrtc_speech_active:

            if not self.silero_working:
                self.silero_working = True

                # Run the intensive check in a separate thread
                threading.Thread(target=self.silero_vad_model.is_voice_active, args=(data,)).start()

    def clear_audio_queue(self):
        """
        Safely empties the audio queue to ensure no remaining audio
        fragments get processed e.g. after waking up the recorder.
        """
        self.audio_buffer.clear()
        try:
            while True:
                self.audio_queue.get_nowait()
        except:
            # PyTorch's mp.Queue doesn't have a specific Empty exception
            # so we catch any exception that might occur when the queue is empty
            pass

    def _audio_queue_size(self):
        """
        Returns the approximate audio queue size.

        :return: Queue size, or None where mp.Queue.qsize() is not implemented (macOS)
        """
        try:
            return self.audio_queue.qsize()
        except NotImplementedError:
            if not getattr(self, "_qsize_unsupported_logged", False):
                logger.warning("Audio queue size is unavailable on this platform; "
                               "buffer overflow handling is skipped.")
                self._qsize_unsupported_logged = True
            return None

    def _is_voice_active(self):
        """
        Determine if voice is active.
        """
        return self.is_webrtc_speech_active and self.is_silero_speech_active

    def _set_state(self, new_state):
        """
        Update the current state of the recorder and execute
        corresponding state-change callbacks.

        Args:
            new_state (str): The new state to set.

        """
        # Check if the state has actually changed
        if new_state == self.state:
            return

        # Store the current state for later comparison
        old_state = self.state

        # Update to the new state
        self.state = new_state

        # Log the state change
        logger.info(f"State changed from '{old_state}' to '{new_state}'")

        # Execute callbacks based on transitioning FROM a particular state
        if old_state == "listening":
            if self.on_vad_detect_stop:
                self.on_vad_detect_stop()

        # Execute callbacks based on transitioning TO a particular state
        if new_state == "listening":
            if self.on_vad_detect_start:
                self.on_vad_detect_start()
            self._set_spinner("speak now")
            if self.spinner and self.halo:
                self.halo._interval = 250
        elif new_state == "transcribing":
            if self.on_transcription_start:
                self.on_transcription_start()
            self._set_spinner("transcribing")
            if self.spinner and self.halo:
                self.halo._interval = 50
        elif new_state == "recording":
            self._set_spinner("recording")
            if self.spinner and self.halo:
                self.halo._interval = 100
        elif new_state == "inactive":
            if self.spinner and self.halo:
                self.halo.stop()
                self.halo = None

    def _set_spinner(self, text):
        """
        Update the spinner's text or create a new
        spinner with the provided text.

        Args:
            text (str): The text to be displayed alongside the spinner.
        """
        if self.spinner:
            # If the Halo spinner doesn't exist, create and start it
            if self.halo is None:
                self.halo = halo.Halo(text=text)
                self.halo.start()
            # If the Halo spinner already exists, just update the text
            else:
                self.halo.text = text

    def _preprocess_output(self, text, preview=False):
        """
        Preprocesses the output text by removing any leading or trailing
        whitespace, converting all whitespace sequences to a single space
        character, and capitalizing the first character of the text.

        Args:
            text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        text = re.sub(r'\s+', ' ', text.strip())

        if text:
            text = text[0].upper() + text[1:]

        # Ensure the text ends with a proper punctuation
        # if it ends with an alphanumeric character
        if not preview:
            if text and text[-1].isalnum():
                text += '.'

        return text

    def __enter__(self):
        return self

    def __exit__(self):
        self.shutdown()
