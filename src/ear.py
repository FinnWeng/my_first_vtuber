#!/usr/bin/env python3

# prerequisites: as described in https://alphacephei.com/vosk/install and also python module `sounddevice` (simply run command `pip install sounddevice`)
# Example usage using Dutch (nl) recognition model: `python test_microphone.py -m nl`
# For more help run: `python test_microphone.py -h`

import argparse
import queue
import sys
import sounddevice as sd

from vosk import Model, KaldiRecognizer, GpuInit
import time


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

class Ear:
    def __init__(self, samplerate = None, device = None, lang = "en-us") -> None:
        
        device_info = sd.query_devices(device, "input")
        if samplerate is None:
            self.samplerate = int(device_info["default_samplerate"])
        else:
            self.samplerate = samplerate
        self.q = queue.Queue()
        model = Model(lang=lang)
        self.rec = KaldiRecognizer(model, self.samplerate)
        self.blank_begin_time = None
        self.blank_limit_time = None
        self.tmp_text = ""
    
    def hear(self):


        data = self.q.get()
        if self.rec.AcceptWaveform(data):

            output_text = self.rec.Result()
            # print(output_text)
            return output_text[14:-3]
        else:

            self.rec.PartialResult()
            return ""
                    
    def hear_call_back(self,indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))








if __name__ == "__main__":

    q = queue.Queue()
    GpuInit()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-l", "--list-devices", action="store_true",
        help="show list of audio devices and exit")
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        "-f", "--filename", type=str, metavar="FILENAME",
        help="audio file to store recording to")
    parser.add_argument(
        "-d", "--device", type=int_or_str,
        help="input device (numeric ID or substring)")
    parser.add_argument(
        "-r", "--samplerate", type=int, help="sampling rate")
    parser.add_argument(
        "-m", "--model", type=str, help="language model; e.g. en-us, fr, nl; default is en-us")
    args = parser.parse_args(remaining)


    if args.samplerate is None:
        device_info = sd.query_devices(args.device, "input")
        # soundfile expects an int, sounddevice provides a float:
        args.samplerate = int(device_info["default_samplerate"])
        
    if args.model is None:
        model = Model(lang="en-us")
    else:
        model = Model(lang=args.model)

    if args.filename:
        dump_fn = open(args.filename, "wb")
    else:
        dump_fn = None
    # import pdb
    # pdb.set_trace()
    
    rec = KaldiRecognizer(model, args.samplerate)

    with sd.RawInputStream(samplerate=args.samplerate, blocksize = 8000, device=args.device,
            dtype="int16", channels=1, callback=callback):
        print("#" * 80)
        print("Press Ctrl+C to stop the recording")
        print("#" * 80)

        
        while True:
            # data = q.get()
            # rec.AcceptWaveform(data)# necessary
            # print("not accept waveform",rec.PartialResult())

            data = q.get()
            if rec.AcceptWaveform(data):
                print("accept waveform",rec.Result())
            else:
                print("not accept waveform",rec.PartialResult())
            if dump_fn is not None:
                dump_fn.write(data)
 
