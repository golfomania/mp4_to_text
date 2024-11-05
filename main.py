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
# import torch # test showed no improvement in speed
# Set the number of threads
# torch.set_num_threads(10)

# ask in CLI for the suffix text
suffix = input("Please enter the suffix of the file you want to transcribe: ")

# Load the Whisper model
# model = whisper.load_model("tiny")
# model = whisper.load_model("tiny.en")
# model = whisper.load_model("base.en")
model = whisper.load_model("turbo")
# model = whisper.load_model("large")

# Record the start time
print("Start decoding the audio file at: ", datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S'))
start_time = time.time()

# Set decoding options with language specified as German
options = {
     # "language":"de", 
     "language":"en", 
     "fp16":False,
     "beam_size": 5 #default: 5
    }

# Decode the audio
result = model.transcribe("output.mp3", **options)
# print(result["text"])

# Save the recognized text
with open(datetime.datetime.now().strftime('%Y.%m.%d') + "_transcription_" + suffix + ".txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = (end_time - start_time) / 60
print(f'\n\nTime needed to run the script: {elapsed_time:.1f} minutes')

# count words
with open(datetime.datetime.now().strftime('%Y.%m.%d') + "_transcription_" + suffix + ".txt", 'r', encoding='utf-8') as file:
    content = file.read()
words = content.split()
word_count = len(words)
print(f'\nThe number of words in the transcription file: {word_count}')