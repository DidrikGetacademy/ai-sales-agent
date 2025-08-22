from langchain_core.tools import tool
from PIL import Image
import requests
from typing import Optional

@tool 
def Text_to_speech(text: str,Audio_speech_folder: str, speaker: Optional[str] = "random") -> str:
    """A tool that receives text as input and outputs a human-voice speech audio file.
       Args:
        text (str): Text that will be converted to a human-voice speech.
        Audio_speech_folder (str): The path to the output folder where the speech file will be saved.
        speaker (str, optional): Speaker identifier for voice selection. Defaults to 'random'.

        Returns:
            str: A confirmation message with the full path where the audio file is stored.
    
    """
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
    proccessor = SpeechT5Processor.from_pretrained(r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained(r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\speecht5_tts")

    inputs = proccessor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_id"], speaker=speaker)
    Audio_speech_folder + f"{text[:10]}.wav"
    with open(Audio_speech_folder, "wb") as f:
        f.write(speech.numpy().tobytes())
    
    final_message = f"A Speech Audio file have been created\n Speech file path: {Audio_speech_folder}"

    return final_message


#--------------------------------------#
#           Custom  Tools
#--------------------------------------#
@tool 
def analyze_vision_speech(processor, prompt: str, audio_path: str, image_url: str):
    """Analyze Sound & Vision"""
    import soundfile
    image = Image.open(requests.get(image_url, stream=True).raw)
    audio = soundfile.read(audio_path)
    inputs = processor(text=prompt, images=[image], audios=[audio], return_tensors='pt').to('cuda:0')
    return inputs 


@tool
def analyze_image(processor, image_url: str):
    """Analyze an image"""
    img = Image.open(requests.get(image_url, stream=True).raw)
    prompt = "What is shown in this image?"
    inputs = processor(text=prompt, images=[img], return_tensor="pt").to("cuda:0")  #using the pipeline chatmodel
    return inputs


@tool 
def transcribe_speech(audio_path: str):
    """Speech to text"""