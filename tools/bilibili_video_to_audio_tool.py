import asyncio
import http
import os
import subprocess
import types
import uuid

import httpx
from bilibili_api import Credential, video, HEADERS

from tools.framework.vocal_separation.video_process import separate_vocals

from .framework.video_to_csv.srt2csv import srt2csv
from .framework.video_to_csv.video_to_text import run_whisper, Video2Subtitles

import pysrt


def generate():
    """
    生成UUID
    """
    uuid_str = str(uuid.uuid4())
    uuid_str = uuid_str.replace("-", "")
    return uuid_str


def extract_audio(mp4_file, output_dir):
    # 生成输出文件路径
    output_file = f"{output_dir}/audio.wav"
    # 使用ffmpeg将视频转换为WAV格式音频
    command = f"ffmpeg -i {mp4_file} -vn -acodec pcm_s16le -ar 44100 -ac 2 {output_file}"
    subprocess.run(command, shell=True)
    return output_file


class BiliBiliVideoDownload:
    def __init__(self):
        return

    def batch_download(self, video_ids: list[str], bilibili_cookie: str, output_path: str):
        file_paths = []
        for video_id in video_ids:
            file_path = self.download(video_id, bilibili_cookie, output_path)
            file_paths.append(file_path)
        return file_paths

    def download(self, video_id: str, bilibili_cookie: str, output_path: str):
        return asyncio.get_event_loop().run_until_complete(self.__download_file(video_id, bilibili_cookie, output_path))

    async def __download_file(self, video_id: str, bilibili_cookie: str, output_path: str):

        # 构建文件路径
        file_path = f"{output_path}/{video_id}.mp4"

        # 解析cookie
        cookie = http.cookies.SimpleCookie()
        cookie.load(bilibili_cookie)
        SESSDATA = cookie.get("SESSDATA").value
        BILI_JCT = cookie.get("bili_jct").value
        BUVID3 = cookie.get("buvid3").value

        # FFMPEG 路径，查看：http://ffmpeg.org/
        FFMPEG_PATH = "ffmpeg"

        # 实例化 Credential 类
        credential = Credential(
            sessdata=SESSDATA, bili_jct=BILI_JCT, buvid3=BUVID3)
        # 实例化 Video 类
        v = video.Video(bvid=video_id, credential=credential)
        # 获取视频下载链接
        download_url_data = await v.get_download_url(0)
        # 解析视频下载信息
        detecter = video.VideoDownloadURLDataDetecter(data=download_url_data)
        streams = detecter.detect_best_streams()
        # 有 MP4 流 / FLV 流两种可能
        if detecter.check_flv_stream() == True:
            # FLV 流下载
            await self.__download_reqeust(streams[0].url, "flv_temp.flv", "FLV 音视频流")
            # 转换文件格式
            os.system(f'{FFMPEG_PATH} -i flv_temp.flv {file_path}')
            # 删除临时文件
            os.remove("flv_temp.flv")
        else:
            # MP4 流下载
            await self.__download_reqeust(streams[0].url, "video_temp.m4s", "视频流")
            await self.__download_reqeust(streams[1].url, "audio_temp.m4s", "音频流")
            # 混流
            os.system(
                f'{FFMPEG_PATH} -i video_temp.m4s -i audio_temp.m4s -vcodec copy -acodec copy {file_path}')
            # 删除临时文件
            os.remove("video_temp.m4s")
            os.remove("audio_temp.m4s")
        return file_path

    async def __download_reqeust(self, url: str, out: str, info: str):
        # 下载函数
        async with httpx.AsyncClient(headers=HEADERS) as sess:
            resp = await sess.get(url)
            length = resp.headers.get('content-length')
            with open(out, 'wb') as f:
                process = 0
                for chunk in resp.iter_bytes(1024):
                    if not chunk:
                        break

                    process += len(chunk)
                    print(f'下载 {info} {process} / {length}')
                    f.write(chunk)


class BiliBiliVideo2AudioTool:
    video_download: BiliBiliVideoDownload
    video2subtitles: Video2Subtitles

    def __init__(self):
        self.video_download = BiliBiliVideoDownload()
        self.video2subtitles = Video2Subtitles()
        return

    def run(self, args):

        video_ids = args["video_ids"]
        video_ids = video_ids.split(",")
        bilibili_cookie = args["bilibili_cookie"]
        output_path = args["output_path"]

        # 批量下载bilibili视频
        video_file_paths = self.video_download.batch_download(
            video_ids, bilibili_cookie, output_path)

        for video_file_path in video_file_paths:

            # 获取MP4名称，不包含扩展名
            basename = os.path.splitext(os.path.basename(video_file_path))[0]
            audio_dir_path = os.path.join(output_path, basename, "audio")

            # 创建目录
            os.makedirs(audio_dir_path, exist_ok=True)

            # 提取音频
            print("======================== 提取 音频 ======================== ")
            audio_file_path = extract_audio(
                video_file_path, audio_dir_path)

            # 人声音频路径和背景音路径
            output_vocal_dir = os.path.join(output_path, basename, "vocal")
            output_instrumental_dir = os.path.join(
                output_path, basename, "inst")

            # 分离人声
            print("======================== 分离 人声 ======================== ")
            separate_vocals(audio_file_path, output_vocal_dir,
                            output_instrumental_dir)