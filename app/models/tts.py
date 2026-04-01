# models/tts.py

import edge_tts
import uuid, os

async def text_to_speech(text):
    filename = f"{uuid.uuid4()}.mp3"
    path = os.path.join("audio", filename)

    communicate = edge_tts.Communicate(
        text,
        voice="en-US-AriaNeural",
        rate="+10%",       # 🔥 faster
        pitch="+2Hz"       # 🔥 more natural
    )

    await communicate.save(path)

    return filename