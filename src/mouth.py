import torch
from transformers import AutoProcessor, BarkModel

from pydub import AudioSegment
from pydub.playback import play
import io

import scipy
import numpy as np

import noisereduce as nr

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained("suno/bark")

    # model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)

    model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(device)

    voice_preset = "v2/en_speaker_9"
    # voice_preset = "v2/hi_speaker_5"
    # voice_preset = "v2/ja_speaker_0"

    text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
    """

    inputs = processor(text_prompt, voice_preset=voice_preset)

    inputs = inputs.to(device)

    audio_array = model.generate(**inputs)

    print(type(audio_array))

    audio_array = audio_array.cpu().numpy().squeeze() # (127680,)

    print(audio_array.dtype)


    sample_rate = model.generation_config.sample_rate


    audio_array = audio_array.astype(np.float32)


    reduced_noise = nr.reduce_noise(y=audio_array, sr=sample_rate)
    audio_segment = AudioSegment(audio_array.tobytes(), 
            frame_rate=sample_rate,
            sample_width=audio_array.dtype.itemsize, 
            channels = len(audio_array.shape))
            
    print("audio_array.dtype.itemsize",audio_array.dtype.itemsize)
    
    play(audio_segment)


    # scipy.io.wavfile.write("./bark_out.wav", rate=sample_rate, data=audio_array)

    # rate, data = scipy.io.wavfile.read("./bark_out.wav")

    # reduced_noise = nr.reduce_noise(y=data, sr=rate)

    # audio_segment = AudioSegment(data.tobytes(), 
    #     frame_rate=rate,
    #     sample_width=data.dtype.itemsize, 
    #     channels = len(data.shape))
    # play(audio_segment)

    '''
    AudioSegment.from_file got some problem
    '''
    
    # audio_segment = AudioSegment.from_file("./bark_out.wav", format="wav")
    # print(audio_segment.frame_rate)
    # # audio_segment.low_pass_filter(5000).high_pass_filter(200)
    # play(audio_segment)



   