############################################
# source: https://github.com/openai/whisper
# source: https://www.youtube.com/watch?v=UAdX0cGuC28&t
# update: pip install -U openai-whisper
############################################
# mp4 to mp3 with PowerShell
# $filepath = "path\to\your\file"
# ffmpeg -i $filepath -q:a 0 -map a ./output.mp3 -y
#
# first section of file (for investigations / tests) with PowerShell
# ffmpeg -i ./output.mp3 -t 00:01:00 -c copy output_short.mp3
#
# check duration of mp3 file with PowerShell
# $output = ffmpeg -i output.mp3 2>&1
# $duration = $output | Select-String -Pattern "Duration" | ForEach-Object { $_.Line.Split(" ")[3].Trim(',') }
# Write-Output $duration
############################################
import datetime
import whisper
import time
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import json
import os
from dotenv import load_dotenv

# Load environment variables at the start of the script
load_dotenv()

# ask in CLI for the suffix text
suffix = input("Please enter the suffix of the file you want to transcribe: ")
selected_language = input("What language was the meeting [0]mixed, [1]english(default), [2]german: ")
if not selected_language:
    selected_language = "1"
selected_model = input("What AI model to use [0]large, [1]turbo(default), [2]base.en, [3]tiny.en, [4]tiny: ")
if not selected_model:
    selected_model = "1"

# Load the Whisper model
match selected_model:
    case "0":
        model = whisper.load_model("large")
    case "1":
        model = whisper.load_model("turbo")
    case "2":
        model = whisper.load_model("base.en")
    case "3":
        model = whisper.load_model("tiny.en")
    case "4":
        model = whisper.load_model("tiny")
    case _:
        model = whisper.load_model("turbo")

# Initialize speaker diarization pipeline
# Get token from environment variable
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("HF_TOKEN not found in environment variables. Please check your .env file.")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token=hf_token
)

# Record the start time
print("\nStart processing the audio file at: ", datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S'))
start_time = time.time()

if selected_language == "0":
    meeting_language = ""
elif selected_language == "1":
    meeting_language = "en"
elif selected_language == "2":
    meeting_language = "de"

# Set decoding options with language specified
options = {
     "language": meeting_language, 
     "fp16": False,
     "beam_size": 5 #default: 5
    }

# Perform speaker diarization
print("Performing speaker diarization...")
diarization = pipeline("output.mp3", num_speakers=2)

# Load audio file
audio = AudioSegment.from_mp3("output.mp3")

# Process transcription with speaker segments
transcription_segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    # Extract segment
    start_time_ms = int(turn.start * 1000)
    end_time_ms = int(turn.end * 1000)
    segment = audio[start_time_ms:end_time_ms]
    
    # Export segment to temporary file
    segment.export("temp_segment.wav", format="wav")
    
    # Transcribe segment
    result = model.transcribe("temp_segment.wav", **options)
    
    # Store segment info
    transcription_segments.append({
        "speaker": speaker,
        "start": turn.start,
        "end": turn.end,
        "text": result["text"].strip()
    })

# Save the recognized text with speaker information
output_filename = datetime.datetime.now().strftime('%Y.%m.%d') + "_transcription_" + suffix + ".txt"
with open(output_filename, "w", encoding="utf-8") as f:
    for segment in transcription_segments:
        f.write(f"[{segment['speaker']}] {segment['text']}\n\n")

# Save detailed JSON output
json_filename = datetime.datetime.now().strftime('%Y.%m.%d') + "_transcription_" + suffix + ".json"
with open(json_filename, "w", encoding="utf-8") as f:
    json.dump(transcription_segments, f, ensure_ascii=False, indent=2)

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = (end_time - start_time) / 60
print(f'Time needed to run the script: {elapsed_time:.1f} minutes')

# count words
with open(output_filename, 'r', encoding='utf-8') as file:
    content = file.read()
words = content.split()
word_count = len(words)
print(f'The number of words in the transcription file: {word_count}')
print(f'Transcription finished for file: {suffix}\n')