from src.llm_engine import LLM_Engine
from src.ear import Ear
import sounddevice as sd





if __name__ == "__main__":
    llm_model_path = "/home/workspace/llava/llava-v1.5-7b/ggml-model-f16.gguf"
    llm_engine = LLM_Engine(llm_model_path, use_own_handler = True, db_path = "./db/")
    ear = Ear()

    # while True:
    #     message = input("enter some text: ")
    #     response = llm_engine.inference(message_input=message)
    #     print(response + "\n")

    ear = Ear()
    with sd.RawInputStream(samplerate=ear.samplerate, blocksize = 8000, device=None,
        dtype="int16", channels=1, callback=ear.hear_call_back):
 
        while True:
            hear_text = ear.hear()
            
            if len(hear_text)>0:
                print(hear_text)
                message = hear_text
                response = llm_engine.inference(message_input=message)
                print(response + "\n")
