# import webrtcvad

# vad = webrtcvad.Vad(2)

# def is_speech_chunk(chunk):
#     frame_size = 320  # 20ms

#     for i in range(0, len(chunk), frame_size):
#         frame = chunk[i:i + frame_size]

#         if len(frame) < frame_size:
#             continue

#         try:
#             if vad.is_speech(frame.tobytes(), 16000):
#                 return True
#         except:
#             continue

#     return False


# # app/utils/audio.py
# import ffmpeg
# import uuid

# def convert_to_wav(input_path):
#     output_path = f"temp_{uuid.uuid4()}.wav"

#     ffmpeg.input(input_path).output(
#         output_path,
#         ar=16000,      # sample rate
#         ac=1           # mono
#     ).run(overwrite_output=True, quiet=True)

#     return output_path

# utils/audio.py
import noisereduce as nr
import numpy as np
import wave
import ffmpeg
import uuid
import os

def convert_to_wav(input_path):
    output_dir = "output_audio"
    os.makedirs(output_dir, exist_ok=True)  # ✅ create directory if not exists

    output_path = f"{output_dir}/temp_{uuid.uuid4()}.wav"

    if not os.path.exists(input_path):
        raise ValueError("Input file not found")

    if os.path.getsize(input_path) == 0:
        raise ValueError("Empty audio file")

    try:
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                format="wav",
                acodec="pcm_s16le",  # ✅ REQUIRED
                ac=1,                # mono
                ar=16000             # 16kHz
            )
            .run(overwrite_output=True)
        )

    except ffmpeg.Error as e:
        print("🔥 FFmpeg ERROR:")
        print(e.stderr.decode() if e.stderr else "No stderr")
        raise RuntimeError("Audio conversion failed")

    return output_path




def reduce_noise(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        rate = wf.getframerate()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    reduced = nr.reduce_noise(y=audio.astype(float), sr=rate)

    output_path = wav_path.replace(".wav", "_clean.wav")

    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(reduced.astype(np.int16).tobytes())

    return output_path

import webrtcvad

vad = webrtcvad.Vad(2)

def is_speech_chunk(audio_bytes):
    sample_rate = 16000
    frame_duration = 20  # ms
    frame_size = int(sample_rate * frame_duration / 1000) * 2  # ✅ 640 bytes

    for i in range(0, len(audio_bytes), frame_size):
        frame = audio_bytes[i:i + frame_size]

        if len(frame) < frame_size:
            continue

        try:
            if vad.is_speech(frame, sample_rate):
                return True
        except Exception:
            continue

    return False



def is_valid_audio(audio_np):
    energy = np.mean(np.abs(audio_np))
    return energy > 200   # threshold