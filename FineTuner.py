from huggingsound import SpeechRecognitionModel, TrainingArguments
from sklearn.model_selection import train_test_split
import os


def fine_tune():
    model_to_train = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-russian", device="cpu")
    output_dir = os.path.join(os.getcwd(), "model_settings", "model_checkpoints")
    audio_data_dir = os.path.join(os.getcwd(), "data", "audio")
    transcript_data_dir = os.path.join(os.getcwd(), "data", "transcript")

    data = []

    # Writing paths to audio and its transcription to train
    for path in os.listdir(transcript_data_dir):
        text = open(os.path.join(transcript_data_dir, path)).read().strip()
        audio_path = path.replace("transcript", "audio").replace("txt", "wav")
        data.append(
            {"path": os.path.join(audio_data_dir, audio_path), "transcription": text})

    X_train, X_test = train_test_split(data, test_size=0.33)

    model_to_train.finetune(
        output_dir,
        num_workers=3,
        train_data=X_train,
        eval_data=X_test,  # the eval_data
        token_set=model_to_train.token_set,
        training_args=TrainingArguments(overwrite_output_dir=True)
    )
    return model_to_train


# Saving
# model_to_train.model.push_to_hub("rmgaliullin/wav2vec2-based-obscenity-detector",
#                                  use_auth_token="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")

if __name__ == "__main__":
    result_model = fine_tune()
    # result_model.model.push_to_hub("rmgaliullin/wav2vec2-based-obscenity-detector",
    #                                use_auth_token="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")
    result_model.evaluate()
