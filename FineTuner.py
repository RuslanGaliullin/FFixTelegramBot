from huggingsound import SpeechRecognitionModel, TrainingArguments
from sklearn.model_selection import train_test_split
import os
import argparse, sys


def load_data(path_to_audio: str, path_to_transcript: str) -> list[dict]:
    audio_data_dir = path_to_audio
    transcript_data_dir = path_to_transcript

    data = []

    # Writing paths to audio and its transcription to train
    for path in os.listdir(transcript_data_dir):
        text = open(os.path.join(transcript_data_dir, path)).read().strip()
        audio_path = path.replace("transcript", "audio").replace("txt", "wav")
        data.append(
            {"path": os.path.join(audio_data_dir, audio_path), "transcription": text})

    return data


def fine_tune(train_data: list[dir], device: str = 'cpu'):
    model_to_train = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-russian", device=device)
    output_dir = os.path.join(os.getcwd(), "model_settings", "model_checkpoints")

    model_to_train.finetune(
        output_dir,
        num_workers=3,
        train_data=train_data,
        token_set=model_to_train.token_set,
        training_args=TrainingArguments(overwrite_output_dir=True)
    )
    return model_to_train


if __name__ == "__main__":
    if len(sys.argv) != 13:
        print(
            "python FineTuner.py --audio --transcript --percent --evaluation --device --push")
        exit(-1)
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio", help="path to dir with audio files")
    parser.add_argument("--transcript", help="path to dir with transcripts")
    parser.add_argument("--percent", help="float value from (0; 1) proportion of data will be used for test")
    parser.add_argument("--evaluation", help="flag 0 or 1 to evaluate result model or not")
    parser.add_argument("--device", help="cpu or cuda device to compute")
    parser.add_argument("--push", help="flag 0 or 1 save weights in hugging face hub or not")
    args = parser.parse_args()
    if args.evaluation == '0':
        X_train, X_test = load_data(args.audio, args.transcript), []
    else:
        X_train, X_test = train_test_split(load_data(args.audio, args.transcript), test_size=float(args.percent))

    result_model = fine_tune(X_train, device=args.device)
    if args.push == '1':
        result_model.model.push_to_hub("rmgaliullin/wav2vec2-based-obscenity-detector",
                                       use_auth_token="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")
    if args.evaluation == '1':
        print(result_model.evaluate(X_test))
