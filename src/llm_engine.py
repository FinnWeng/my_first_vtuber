from llama_cpp import Llama

# llm_model_path = "/home/workspace/llava/llava-v1.5-7b/ggml-model-f16.gguf"
# llm = Llama(model_path=llm_model_path,n_gpu_layers=-1,)
# output = llm(
#       "Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)

# print(output)

class LLM_Engine:
    def __init__(self, llm_model_path, ) -> None:
        self.llm = Llama(model_path = llm_model_path,
                         n_gpu_layers=-1,
                         )
    
    def inference(self, message):
        output = self.llm(message, max_tokens=32, stop=["Q:", "\n"], echo=True)
        message_length = len(message)
        # import pdb
        # pdb.set_trace()
        response = output["choices"][0]["text"][message_length:]
        return response
