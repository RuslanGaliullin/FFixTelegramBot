import random

from huggingsound import SpeechRecognitionModel, TrainingArguments

from transformers import AutoModelForCTC, Wav2Vec2Processor
import os

model_to_train = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-russian", device="cpu")
output_dir = os.path.join(os.getcwd(), "model_settings", "model_checkpoints")
audio_data_dir = os.path.join(os.getcwd(), "data", "audio")
transcript_data_dir = os.path.join(os.getcwd(), "data", "transcript")
# first of all, you need to define your model's token set
# however, the token set is only needed for non-finetuned models
# if you pass a new token set for an already finetuned model, it'll be ignored during training

# define your train/eval data

train_data = []

for path in os.listdir(transcript_data_dir):
    text = open(os.path.join(transcript_data_dir, path)).read()
    audio_path = path.replace("transcript", "audio").replace("txt", "wav")
    train_data.append(
        {"path": os.path.join(audio_data_dir, audio_path), "transcription": text})

# and finally, fine-tune your model
model_to_train.finetune(
    output_dir,
    num_workers=3,
    train_data=train_data,
    eval_data=[random.choice(train_data) for i in range(10)],  # the eval_data
    token_set=model_to_train.token_set,
    training_args=TrainingArguments(overwrite_output_dir=True)
)
# print(model_to_train.transcribe(["/home/user/FFixTelegramBot/data/audio/audio_25.wav"])[0]["transcription"])
# model_to_train.trainer.save_model("rmgaliullin/wav2vec2-based-obscenity-detector",
#                                   use_auth_token="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")
model_to_train.model.push_to_hub("rmgaliullin/wav2vec2-based-obscenity-detector",
                                 use_auth_token="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")
