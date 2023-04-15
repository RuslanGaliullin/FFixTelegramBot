from huggingsound import SpeechRecognitionModel
from transformers import AutoModelForCTC, Wav2Vec2Processor

detector = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
# detector.finetune()
detector.model.push_to_hub("rmgaliullin/wav2vec2-based-obscenity-detector")
smth = "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"
detector.processor.push_to_hub("rmgaliullin/wav2vec2-based-obscenity-detector", use_auth_token=smth)
