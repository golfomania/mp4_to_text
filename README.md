# mp4_to_text

using AI to convert mp4>mp3>transcription>translation

## idea

the idea is to transcribe the audio of a video file (e.g. from a meeting recording) to text and then translate the text to another language, all using local running code and AI models.

## mp4 to mp3

- using ffmpeg
- ffmpeg -c:a copy
- ffmpeg -i input.mp4 -q:a 0 -map a output.mp3

## mp3 to text

using openAI Whisper

## text to translation

translation to english is included in the openAI Whisper

## sum up

creating a summary of the meeting, using AI
Test done with Copilot by uploading local txt file and ask for a summary > worked

# transcription speed

test for a 1min mp3 file (4700 words) on a X1 Gen8 laptop (i7 10 core 1,7MHz / 32GB RAM)

- tiny 0.1min
- turbo 0.7min
- large

## ToDo

- [ ] summarize the transcription locally
- [ ] speaker recognition
- [ ] translation to other languages
