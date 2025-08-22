import torch
from langchain_huggingface import  ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from typing import TypedDict, List
from transformers import GenerationConfig
###Rag Arkitektur er viktigere en model kompalitet#####
model_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\Phi-4-multimodal-instruct"

#-------------------------------------#
#Quantization for efficiency
#--------------------------------------#
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)

#---------------------------------------#
#       Model Loading
#---------------------------------------#
pipeline = HuggingFacePipeline.from_model_id(
    model_id=model_path,
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 1000, "do_sample": False, "repetition_penalty":1.03},
    model_kwargs={"trust_remote_code": True, "quantization_config": quantization_config, "device_map": "auto"}

)


#---------------------------------------#
#       Wrapping as chatmodel
#---------------------------------------#
chat_model = ChatHuggingFace(llm=pipeline)



#--------------------------------------#
#  custom processor for multimodality
#--------------------------------------#
processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)



#-----------------------------------------#
#  Generation Config for phi-4 multimodel
#------------------------------------------#
generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')



#--------------------------------------#
#       Agent message state
#--------------------------------------#
class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    images: List[str]
    Audio: List[str]



#--------------------------------------#
#    A node that listen Messages
#--------------------------------------#
def agent_node(state: AgentState):
    content = [{"type": "text", "text": state["messages"[-1]].content}]

    if state.get("images"):
        content.extend([{"type": "image_url", "image_url": {"url": img}} for img in state["images"]])

    if state.get("Audio"):
        content.extend([{"type": "audio_url", "audio_url": {"url": audio}} for audio in state["audio"]])

    response = chat_model.invoke([HumanMessage(content=content)])
    return {"messages": state["messages"] + [response]}



#--------------------------------------#
#    A node that listen for tool calls
#--------------------------------------#
def tool_node(state: AgentState):
    from Custom_tools import analyze_image,Text_to_speech,analyze_vision_speech,transcribe_speech
    if "analyze_image" in state["messages"][-1].content:
        result = analyze_image.invoke({"image_url": state["images"][0]})
        return {"messages": state["messages"] + [AIMessage(content=result)]}
    
    if "Text_to_speech" in state["messages"][-1].content:
        result = Text_to_speech.invoke({""})
        return
    
    if "analyze_vision_speech" in state["messages"][-1].content:
        result = analyze_vision_speech.invoke({""})
        return
    
    if "transcribe_speech" in state["messages"][-1].content:
        result = transcribe_speech.invoke({""})
        return
    
    return state



#--------------------------------------#
#    Build Workflow/graph TREE
#--------------------------------------#
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge("agent", "tools")
workflow.add_edge("tools",END)
graph = workflow.compile()

inputs = {"messages": [HumanMessage(content="What is shown in this image?")], "images": ["https://www.ilankelman.org/stopsigns/australia.jpg"]}
result = graph.invoke(inputs)
print(result["messages"][-1].content)

inputs["messages"].append(HumanMessage(content="What is special about it?"))
result = graph.invoke(inputs)
print(result["messages"][-1].content) 