from speech_to_text_whisper import AudioToTextRecorder


def process_text(text):
    print(text)

if __name__ == '__main__':
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder(api_connection=False)

    while True:
        recorder.text(process_text)


#We would like to inform you that the issue related to the Contract ID has been successfully resolved across all environments: Development, QA, and Production.

"""
API connection = false, model = tiny
DEBUG - Model completed transcription in 0.84 seconds
INFO - Text after transcription: We would like to inform you that the issue related to the contact ID has been successfully resolved across all the environments developed in QA and production.; Unknown Speaker (Best Match: oleh, Score: 0.6028)
INFO - Memory used for transcription: 251.12 MB

API connection = false, model = large-2
DEBUG - Model completed transcription in 26.89 seconds
INFO - Text after transcription: We would like to inform you that the issue related to the contact ID has been successfully resolved across all environments, development, QA, and production.; Unknown Speaker (Best Match: oleh, Score: 0.5574)
INFO - Memory used for transcription: 260.82 MB

API connection = true, model = whisper-1, lang=None
DEBUG - Model completed transcription in 1.64 seconds
INFO - Text after transcription: We would like to inform you that the issue related to the contact IT has been successfully resolved across all environments, development, QA and production.; Unknown Speaker (Best Match: oleh, Score: 0.5823)
INFO - Memory used for transcription: 243.13 MB

API connection = true, model = whisper-1, lang=en
DEBUG - Model completed transcription in 1.49 seconds
INFO - Text after transcription: We would like to inform you that the issue related to the contact ID has been successfully resolved across all environments, development, QA and production.; Unknown Speaker (Best Match: oleh, Score: 0.4931)
INFO - Memory used for transcription: 245.25 MB
"""

#whisper-base: This is the smallest and least computationally intensive model, and often used for quick tasks or when hardware resources are limited.
#whisper-small: A bit larger and more accurate than the base model, with a higher computational cost.

"""
API connection = false, model = tiny
DEBUG - Model completed transcription in 0.74 seconds
INFO - Text after transcription: This provides this as a smallest and lift computationally intensive model and often used for quick task over hardware resources limited.; Unknown Speaker (Best Match: oleh, Score: 0.5306)
INFO - Memory used for transcription: 257.52 MB
DEBUG - Model completed transcription in 0.66 seconds
INFO - Text after transcription: Whisper small, a bit larger and more accurate sets a base model with a high-competition cost.; Unknown Speaker (Best Match: oleh, Score: 0.5324)
INFO - Memory used for transcription: 245.32 MB

API connection = false, model = large-2
DEBUG - Model completed transcription in 24.14 seconds
INFO - Text after transcription: This is the smallest and least computationally intensive model and often used for quick tasks or when hardware resources are limited.; Unknown Speaker (Best Match: oleh, Score: 0.6134)
INFO - Memory used for transcription: 259.28 MB
DEBUG - Model completed transcription in 24.78 seconds
INFO - Text after transcription: Whisper small, a bit larger and more accurate than the base model with a higher computational cost.; Unknown Speaker (Best Match: oleh, Score: 0.6296)
INFO - Memory used for transcription: 235.49 MB

API connection = true, model = whisper-1, lang=None
DEBUG - Model completed transcription in 1.19 seconds
INFO - Text after transcription: This is the smallest and least computationally intensive model, and often used for quick tasks when hardware resources are limited.; Unknown Speaker (Best Match: oleh, Score: 0.5442)
INFO - Memory used for transcription: 251.93 MB
DEBUG - Model completed transcription in 1.05 seconds
INFO - Text after transcription: Whisper small a bit larger and more accurate than the base model with a higher computational cost; Unknown Speaker (Best Match: oleh, Score: 0.5752)
INFO - Memory used for transcription: 228.25 MB

API connection = true, model = whisper-1, lang=en
DEBUG - Model completed transcription in 1.47 seconds
INFO - Text after transcription: This is the smallest and least computationally intensive model and often used for quick tasks or when hardware resources are limited.; Unknown Speaker (Best Match: oleh, Score: 0.5513)
INFO - Memory used for transcription: 227.18 MB
DEBUG - Model completed transcription in 1.51 seconds
INFO - Text after transcription: Whisper small, a bit larger and more accurate than the base model, with a higher computational cost.; Unknown Speaker (Best Match: oleh, Score: 0.5979)
INFO - Memory used for transcription: 238.32 MB
"""