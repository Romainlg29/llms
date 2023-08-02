import torch
import time
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from optimum.bettertransformer import BetterTransformer

# Define the device to run on
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "openai/whisper-large-v2"

# Load the model
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
model = BetterTransformer.transform(model, keep_original_model=False)
model.config.forced_decoder_ids = None

# Load the processor
processor = WhisperProcessor.from_pretrained(model_name)

# Load the audio file
data, samplerate = sf.read("files/bark_gen_en.wav")

# Process the audio file
inputs = processor(data, sampling_rate=16000, return_tensors="pt").to(device).input_features

# Generate the transcription
predicted_ids = model.generate(inputs, max_new_tokens=600)

# Decode the transcription
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription)