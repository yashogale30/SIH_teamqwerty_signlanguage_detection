from gtts import gTTS
import io
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Text and language
text = "let's meet tomorrow"
language = "en"

# Create gTTS object
speech = gTTS(text=text, lang=language, slow=False, tld="com.au")

# Save to in-memory buffer
buffer = io.BytesIO()
speech.write_to_fp(buffer)

# Move buffer position to the beginning
buffer.seek(0)

# Load the buffer into pygame and play it directly
pygame.mixer.music.load(buffer, "mp3")
pygame.mixer.music.play()

# Keep the script running while the audio plays
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
