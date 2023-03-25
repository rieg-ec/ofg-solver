#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_video.mp4 output_directory"
    exit 1
fi

input_video="$1"
output_directory="$2"

# Create the output directory if it doesn't exist
mkdir -p "${output_directory}"

# Extract the audio as an MP3 file
audio_file="${output_directory}/audio.mp3"
ffmpeg -i "${input_video}" -vn -acodec libmp3lame -ac 2 -ab 192k -ar 48000 "${audio_file}"

# Split the audio file into 5-minute long chunks
ffmpeg -i "${audio_file}" -f segment -segment_time 300 -c copy "${output_directory}/chunk_%03d.mp3"
