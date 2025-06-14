#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
__ToDo："transcribe video to srt via OpenAI Whisper "
__info:"ASR + simplied chinese + noise reduced"
__author: "Aria:(https://github.com/ariafyy)"
"""

import argparse
import os
import pathlib
import torch
from typing import Iterator, TextIO

try:
    import whisper
except ImportError:
    print("check requirements: video_to_text/requirements_run_whisper.txt")
    raise

# 更安全的设备初始化
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

from hanziconv import HanziConv
from subprocess import CalledProcessError, run
import numpy as np

SAMPLE_RATE = 16000
TRANSCRIBE_MODE = ' '  # TRANSCRIBE_MODE = 'noisereduce'


class Video2Subtitles(object):
    def __init__(self):
        MODEL_WHISPER = "small"  # or your local model_path
        WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]
        print("---loading model in your local path or downloading now---")
        try:
            self.model = whisper.load_model(MODEL_WHISPER, device=DEVICE)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def srt_format_timestamp(self, seconds: float):
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        return (f"{hours}:") + f"{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def write_srt(self, transcript: Iterator[dict], file: TextIO):
        count = 0
        for segment in transcript:
            count += 1
            print(
                f"{self.srt_format_timestamp(segment['start'])} --> {self.srt_format_timestamp(segment['end'])}\n"
                f"{self.trad2simp(segment['text']).replace('-->', '->').strip()}\n",
                file=file,
                flush=True,
            )

    def trad2simp(self, text):
        """
        # traditional chinese into simplified chinese
        :param text: 
        :return: 
        """
        simp = HanziConv.toSimplified(text)
        return simp

    def transcribe(self, input_video: str, srt_folder: str):
        subtitle_format = "srt"
        lang = "zh"
        verbose = True
        input_video_ = input_video if isinstance(input_video, str) else input_video.name
        try:
            if TRANSCRIBE_MODE == 'noisereduce':
                audio = self.audio_denoise(input_video_)
            else:
                audio = input_video_
            
            result = self.model.transcribe(
                audio=audio,
                task="transcribe",
                language=lang,
                verbose=verbose,
                initial_prompt=None,
                word_timestamps=True,
                no_speech_threshold=0.95,
                fp16=DEVICE.type == "cuda"
            )
            
            os.makedirs(srt_folder, exist_ok=True)
            subtitle_file = srt_folder + "/" + pathlib.Path(input_video).stem + "." + subtitle_format
            if subtitle_format == "srt":
                with open(subtitle_file, "w") as srt:
                    self.write_srt(result["segments"], file=srt)
            print("\nsubtitle_file:", subtitle_file, "\n")
            return subtitle_file
        except Exception as e:
            print(f"Error during transcription: {e}")
            raise

    def load_audio(self, file: str, sr: int = SAMPLE_RATE):
        """
        Requires the ffmpeg CLI in PATH.
        fmt: off
        """
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]
        # fmt: on
        try:
            out = run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
        data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        return data

    def audio_denoise(self, input_audio: str, ):
        """
        # reduce noise
        """
        try:
            import noisereduce as nr
        except ImportError:
            print("pip install noisereduce")
        rate = SAMPLE_RATE
        data = self.load_audio(input_audio)
        reduced_audio = nr.reduce_noise(y=data, sr=rate)
        return reduced_audio


def run_whisper(video2subtitles: Video2Subtitles, args):
    try:
        if args.verbose:
            print('running whisper')

        # checking if input_video is a file
        if not os.path.isfile(args.input_video):
            print('input_video is not exist')
            return

        # checking if srt_folder is a folder
        if not os.path.isdir(args.srt_folder):
            print('warning srt_folder is not exist')
            # create srt_folder
            os.makedirs(args.srt_folder, exist_ok=True)
            print('create folder', args.srt_folder)

        # run whisper
        input_file = args.input_video
        srt_folder = args.srt_folder
        result = video2subtitles.transcribe(input_file, srt_folder)
        return result
    except Exception as e:
        print(f"Error in run_whisper: {e}")
        raise
