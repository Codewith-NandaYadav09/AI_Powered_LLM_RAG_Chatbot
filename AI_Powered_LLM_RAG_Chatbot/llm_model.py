from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class HuggingFaceLLM:
    def __init__(self, model_name="tiiuae/falcon-7b-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
    
    def generate(self, prompt):
        response = self.pipeline(prompt)[0]['generated_text']
        # Extract only the newly generated text
        return response[len(prompt):].strip()
