from src.llm_engine import LLM_Engine





if __name__ == "__main__":
    llm_model_path = "/home/workspace/llava/llava-v1.5-7b/ggml-model-f16.gguf"
    llm_engine = LLM_Engine(llm_model_path, use_own_handler = True, db_path = "./db/")

    while True:
        message = input("enter some text: ")
        response = llm_engine.inference(message_input=message)
        print(response + "\n")