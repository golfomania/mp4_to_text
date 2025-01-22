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
import warnings
warnings.filterwarnings("ignore", message=".*You are using `torch.load` with `weights_only=False`.*")

# ask in CLI for the suffix text
suffix = input("Please enter the suffix of the file you want to transcribe: ")
# ask for the language of the meeting
selected_language = input("What language was the meeting [0]mixed, [1]english(default), [2]german: ")
if not selected_language:
    selected_language = "1"
if selected_language == "0":
    meeting_language = ""
elif selected_language == "1":
    meeting_language = "en"
elif selected_language == "2":
    meeting_language = "de"
print(f"Selected language: {meeting_language}")
# ask for the AI model to use
selected_model = input("What AI model to use [0]large, [1]turbo(default), [2]base.en, [3]tiny.en, [4]tiny: ")
# Load the Whisper model
match selected_model:
    case "0":
        print("Loading large model")
        model = whisper.load_model("large")
    case "1":
        print("Loading turbo model")
        model = whisper.load_model("turbo")
    case "2":
        print("Loading base.en model") 
        model = whisper.load_model("base.en")
    case "3":
        print("Loading tiny.en model")
        model = whisper.load_model("tiny.en")
    case "4":
        print("Loading tiny model")
        model = whisper.load_model("tiny")
    case _:
        print("Loading turbo model")
        model = whisper.load_model("turbo")

# Record the start time
print("\nStart decoding the audio file at: ", datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S'))
start_time = time.time()

# Set decoding options with language specified as German
options = {
     "language": meeting_language, 
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
print(f'Time needed to run the script: {elapsed_time:.1f} minutes')

# count words
with open(datetime.datetime.now().strftime('%Y.%m.%d') + "_transcription_" + suffix + ".txt", 'r', encoding='utf-8') as file:
    content = file.read()
words = content.split()
word_count = len(words)
print(f'The number of words in the transcription file: {word_count}')
print(f'Transcription finished for file: {suffix}\n')