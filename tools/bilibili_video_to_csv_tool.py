import asyncio
import http
import os
import traceback
import types
import uuid

import httpx
from bilibili_api import Credential, video, HEADERS

from .framework.video_to_csv.srt2csv import srt2csv
from .framework.video_to_csv.video_to_text import run_whisper, Video2Subtitles


def generate():
    """
    生成UUID
    """
    uuid_str = str(uuid.uuid4())
    uuid_str = uuid_str.replace("-", "")
    return uuid_str


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
        credential = Credential(sessdata=SESSDATA, bili_jct=BILI_JCT, buvid3=BUVID3)
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


class BiliBiliVideo2CsvTool:
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
        video_file_paths = self.video_download.batch_download(video_ids, bilibili_cookie, output_path)

        # 批量将视频转换为字幕文件
        srt_file_paths = []
        for video_file_path in video_file_paths:
            args = {
                "verbose": "verbose",
                "input_video": video_file_path,
                "srt_folder": output_path
            }
            args = types.SimpleNamespace(**args)
            srt_file_path = run_whisper(self.video2subtitles, args)
            srt_file_paths.append(srt_file_path)

        # 批量将字幕文件转换为csv
        for srt_file_path in srt_file_paths:
            args = {
                "verbose": "verbose",
                "input_srt": srt_file_path,
                "srt_folder": output_path
            }
            args = types.SimpleNamespace(**args)

            try:
                srt2csv(args)
            except:
                traceback.print_exception()


if __name__ == '__main__':
    args = {
        "video_ids": "BV1AC4y1U77e",
        "bilibili_cookie": "buvid4=34DB59A6-47D1-B3DC-953E-0E39DB1D2E1655624-023101921-ksPq5hUPXVm6%2BOFFmBjPRg%3D%3D; buvid_fp_plain=undefined; enable_web_push=DISABLE; header_theme_version=CLOSE; CURRENT_BLACKGAP=0; _uuid=B43613CB-D72B-839F-2108E-E7962CD10BBAC03668infoc; hit-dyn-v2=1; home_feed_column=5; buvid3=A0E99412-2F6B-9261-C5A6-EFB2D4568E2F97438infoc; b_nut=1732413697; rpdid=|(umu)ul|lml0J'u~JkRuJ)m); DedeUserID=382957163; DedeUserID__ckMd5=537234e3cc45dfda; LIVE_BUVID=AUTO1017331504227621; fingerprint=65a3553cbda12db958eb605567d21737; buvid_fp=65a3553cbda12db958eb605567d21737; CURRENT_QUALITY=80; enable_feed_channel=ENABLE; SESSDATA=a8cc00fd%2C1764736607%2C5b90f%2A62CjA9HNEB_t4ir2-kumXomLTuY_ZwEU5UeKdw7Afb7Mlq-6o0fKxpVdG_IxD0JCJcSGMSVm10OTloMUpjb1haaV9Ga2lpMk9wS0lWd29BeHJyckkxZnFxLThRUTJEcng0M0pHS2NqaS1hNkc3OVNPekhvMHB4WHA0VkUwQU5jclZXYVVuaXpDMU9nIIEC; bili_jct=569089d70a862e0b6546464f6b5eb800; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NDk1NjE2MDcsImlhdCI6MTc0OTMwMjM0NywicGx0IjotMX0.SKAaOBJkCCXlq4v8py2nudszbObC93cHVYOzrMHaD8s; bili_ticket_expires=1749561547; sid=6uoltlfo; b_lsid=7D1EEAEA_1974E95B17D; PVID=1; bp_t_offset_382957163=1076035035888353280; CURRENT_FNVAL=4048; browser_resolution=1680-463",
        "output_path": "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/output"
    }
    video2csv = BiliBiliVideo2CsvTool()
    video2csv.run(args)
