from llama_cpp import Llama

# llm_model_path = "/home/workspace/llava/llava-v1.5-7b/ggml-model-f16.gguf"
# llm = Llama(model_path=llm_model_path,n_gpu_layers=-1,)
# output = llm(
#       "Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)

# print(output)

from llama_cpp._utils import suppress_stdout_stderr
import llama_cpp.llama_types as llama_types
import llama_cpp.llama as llama
import llama_cpp.llama_grammar as llama_grammar
from llama_cpp.llama_chat_format import _convert_completion_to_chat

import json
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Protocol



# def _get_system_message(
#     messages: List[llama_types.ChatCompletionRequestMessage],
# ) -> str:
#     """Get the first system message."""
#     for message in messages:
#         if message["role"] == "system":
#             return message["content"] or ""
#     return ""

class My_Llama_Chat_Handler:
    _clip_free = None

    def __init__(self, verbose: bool = False):
        # import llama_cpp.llava_cpp as llava_cpp

        # self._llava_cpp = llava_cpp
        # self.clip_model_path = clip_model_path
        self.verbose = verbose
        # self._clip_free = self._llava_cpp._libllava.clip_free  # type: ignore


    def __call__(
        self,
        *,
        llama: llama.Llama,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        response_format: Optional[
            llama_types.ChatCompletionRequestResponseFormat
        ] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[llama.LogitsProcessorList] = None,
        grammar: Optional[llama.LlamaGrammar] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]:


        # system_prompt = _get_system_message(messages)
        # system_prompt = (
        #     system_prompt
        #     if system_prompt != ""
        #     else "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and polite answers to the human's questions."
        # )

   
        system_prompt = "A chat between a curious human_creator and an artificial intelligence virtual_youtuber."\
            "The name of AI virtual_youtuber is Homo-chan."\
            "She is not a assitant. And she prefer chatting with people rather than assisting others."\
            "Homo-chan likes singing and dancing and chat with others."\
                "The virtual_youtuber got short-term memory information to help her to response."
        

        user_role = "\nhuman_creator:"
        assistant_role = "\nvirtual_youtuber:"
        llama.reset()
        llama.eval(llama.tokenize(system_prompt.encode("utf8"), add_bos=True))
        for message in messages:
            if message["role"] == "human_creator" and message["content"] is not None:
                if isinstance(message["content"], str):
                    llama.eval(
                        llama.tokenize(
                            f"{user_role} {message['content']}".encode("utf8"),
                            add_bos=False,
                        )
                    )
            if message["role"] == "virtual_youtuber" and message["content"] is not None:
                llama.eval(
                    llama.tokenize(
                        f"Vtuber: {message['content']}".encode("utf8"), add_bos=False
                    )
                )
                assert llama.n_ctx() >= llama.n_tokens
        llama.eval(llama.tokenize(f"{assistant_role}".encode("utf8"), add_bos=False))
        assert llama.n_ctx() >= llama.n_tokens

        prompt = llama.input_ids[: llama.n_tokens].tolist()

        if response_format is not None and response_format["type"] == "json_object":
            try:
                # create grammar from json schema
                if "schema" in response_format:
                    grammar = llama_grammar.LlamaGrammar.from_json_schema(
                        json.dumps(response_format["schema"])
                    )
            except Exception as e:
                grammar = llama_grammar.LlamaGrammar.from_string(llama_grammar.JSON_GBNF)

        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
            ),
            stream=stream,
        )


class LLM_Engine:
    def __init__(self, llm_model_path, use_own_handler = True) -> None:
        if use_own_handler:
            chat_handler = My_Llama_Chat_Handler()
            self.llm = Llama(model_path = llm_model_path,
                            n_gpu_layers=-1,
                            chat_handler = chat_handler,
                            n_ctx=4096, 
                            # verbose=False,
                            )
        else:
            self.llm = Llama(model_path = llm_model_path,
                n_gpu_layers=-1,
                chat_format="chatml",
                n_ctx=4096, 
                # verbose=False,
                )

        # self.role = "system"
        self.short_term_memories = [""]
        self.short_term_memories_max_size = 10
    
    def get_memories(self):
        out = "<<Short-Term Momory Chounk Start>>"
        for i, short_term_memory in enumerate(self.short_term_memories):
            this_dialogue = "{}. {}".format(i, short_term_memory) + ", "
            out = out + this_dialogue
        out = out + "<<Short-Term Momory Chounk End>>"
     
        return out
    
    def save_dialogue(self, message, response):
        dialogue = "Text1(human_creator):{}, Text2(virtual_youtuber):{}".format(message,response)
        return dialogue

    def short_term_memory_append(self, message_input, response):
        this_dialogue = self.save_dialogue(message_input, response)
        self.short_term_memories.append(this_dialogue)
        if len(self.short_term_memories) > self.short_term_memories_max_size:
            self.short_term_memories.pop(0)

    
    def inference(self, message_input):
        
        # message_list = [
        #   {"role": self.role, "content": "You are an assistant who perfectly describes images."},
        #   {
        #       "role": "user",
        #     #   "content": message,
        #       "content": message

        #   }
        # ]
        # output = self.llm.create_chat_completion(messages = message_list)
        # message_length = len(message)
        # print(output)
   
        # response = output["choices"][0]["message"]["content"]

        # output = self.llm(message, max_tokens=32, stop=["Q:", "\n"], echo=True)
        # response = output["choices"][0]["text"]

        message_list =[
            {
                "role": "virtual_youtuber",
                "content": "The short-term memory: " + self.get_memories(),
                
            },
            {"role": "human_creator", "content": message_input},
        ]
        print(self.get_memories())

        # vtb_response_format={
        # "type": "json_object",
        # "schema": {
        #     "type": "object",
        #     "chat_content": {"reponse": {"type": "string"}},
            
        # },
        # }

        output = self.llm.create_chat_completion(
            messages = message_list,
            # response_format = vtb_response_format,
            temperature=0.7,)
        
        # print(output)
        # import pdb
        # pdb.set_trace()
        

        
        response = output["choices"][0]["message"]["content"]

        self.short_term_memory_append(message_input=message_input, response=response)

        

        return response
