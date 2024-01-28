from llama_cpp import Llama

# llm_model_path = "/home/workspace/llava/llava-v1.5-7b/ggml-model-f16.gguf"
# llm = Llama(model_path=llm_model_path,n_gpu_layers=-1,)
# output = llm(
#       "Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)

# print(output)

class LLM_Engine:
    def __init__(self, llm_model_path, role = "system") -> None:
        self.llm = Llama(model_path = llm_model_path,
                         n_gpu_layers=-1,
                         chat_format="llama-2",
                        # n_ctx=2048, verbose=False,
                         )
        self.role = role
    
    def inference(self, message):
        
        message_list = [
          {"role": self.role, "content": "You are an assistant who perfectly describes images."},
          {
              "role": "user",
            #   "content": message,
              "content": message

          }
        ]
        output = self.llm.create_chat_completion(messages = message_list)
        message_length = len(message)
        print(output)
   
        response = output["choices"][0]["message"]["content"]

        # output = self.llm(message, max_tokens=32, stop=["Q:", "\n"], echo=True)
        # response = output["choices"][0]["text"]

        return response
