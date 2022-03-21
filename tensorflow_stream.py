
from __future__ import print_function
import argparse
import ffmpeg
import logging
import numpy as np
import os
import subprocess
import zipfile
import cv2


parser = argparse.ArgumentParser(description='Example streaming ffmpeg numpy processing')
parser.add_argument('in_filename', help='Input filename')
parser.add_argument('out_filename', help='Output filename')
parser.add_argument('in_audio', help='in audio')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_video_size(filename):
    logger.info('Getting video size for {!r}'.format(filename))
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height

def start_ffmpeg_audio_write(fn):
    stream = ffmpeg.input(fn).audio
    output = ffmpeg.output(stream, 'xxccc.wav').overwrite_output()
    print('Will run command:')
    print(ffmpeg.compile(output))

    process = output.run_async(pipe_stdout=True, pipe_stderr=True)
    stdout, stderr = process.communicate(input)
    retcode = process.poll()

def start_ffmpeg_audio_read(fn):
    logger.info('Starting ffmpeg audio')
    # this quietly dumps the right channel into a numpy array
    out, _ = (ffmpeg
             .input(fn)
             .output('pipe:', format='s16le', acodec='pcm_s16le', af='pan=mono|FC=FR')
             .global_args("-loglevel", "quiet")
             .global_args("-nostats")
             .global_args("-hide_banner")
             .run(capture_stdout=True))
    # data = np.frombuffer(out, np.int16)

    return out

def start_ffmpeg_vid_read(in_filename):
    logger.info('Starting ffmpeg process1')
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)

def start_ffmpeg_vid_writer(out_filename, width, height):
    logger.info('Starting ffmpeg process2')
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(out_filename, pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)

def read_vid_frame(process1, width, height):
    logger.debug('Reading frame')
    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
        cv2.imshow('rtspin', frame)
        key = cv2.waitKey(1)
    return frame


def process_vid_frame_simple(frame):
    '''Simple processing example: darken frame.'''
    return frame / 0.78

def write_vid_frame(process2, frame):
    logger.debug('Writing frame')
    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )
    # frame2 = np.frombuffer(frame, dtype=np.uint8).reshape(240, 240, 3)
    cv2.imshow('rtspout', frame)
    key = cv2.waitKey(1)


def run_pipe(in_filename, out_filename, in_audio, process_frame):
    width, height = get_video_size(in_filename)
    vid_read_p = start_ffmpeg_vid_read(in_filename)
    vid_write_p = start_ffmpeg_vid_writer(out_filename, width, height)
    while True:
        in_frame = read_vid_frame(vid_read_p, width, height)
        if in_frame is None:
            logger.info('End of input stream')
            break
        logger.debug('Processing frame')
        out_frame = process_frame(in_frame) # ai model
        write_vid_frame(vid_write_p, out_frame)
    logger.info('Waiting for ffmpeg process1')
    vid_read_p.wait()

    logger.info('Waiting for ffmpeg process2')
    vid_write_p.stdin.close()
    vid_write_p.wait()
    logger.info('Done')


if __name__ == '__main__':
    args = parser.parse_args()
    process_frame = process_vid_frame_simple
    run_pipe(args.in_filename, args.out_filename, args.in_audio, process_frame)
