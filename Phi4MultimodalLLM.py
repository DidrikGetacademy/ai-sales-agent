from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import Generation, LLMResult
from langchain_core.messages import BaseMessage
import torch
from typing import Optional,List
class Phi4MultimodalLLm(BaseLanguageModel):
    def __init__(self, model, processor, generation_config):
        self.model = model,
        self.processor = processor
        self.generation_config = generation_config

    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None):
        