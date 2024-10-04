############################################
# source: https://github.com/openai/whisper
# source: https://www.youtube.com/watch?v=UAdX0cGuC28&t
# update: pip install -U openai-whisper
#
# mp4 to mp3
# ffmpeg -i "C:\Users\B25\OneDrive - Nürnberger Baugruppe GmbH + Co KG\Bilder\Screenpresso\2024-09-30_13h58_06.mp4" -q:a 0 -map a ./output.mp3

# first section of file
# ffmpeg -i ./output.mp3 -t 00:01:00 -c copy output_short.mp3
############################################
import whisper
import time

# Load the Whisper model
# model = whisper.load_model("tiny")
# model = whisper.load_model("turbo")
model = whisper.load_model("large")

# Record the start time
start_time = time.time()

# Set decoding options with language specified as German
options = {
    "language":"de", 
    }

# Decode the audio
result = model.transcribe("output_short.mp3", **options)
print(result["text"])
# Save the recognized text
with open("transcription_2.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = (end_time - start_time) / 60
print(f'\n\nTime needed to run the script: {elapsed_time:.1f} minutes')

# count words
with open('transcription.txt', 'r', encoding='utf-8') as file:
    content = file.read()
words = content.split()
word_count = len(words)
print(f'\nThe number of words in the file is: {word_count}')