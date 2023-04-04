import os

from huggingsound import SpeechRecognitionModel

# model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")

# import librosa
# import soundfile as sf
# from transformers import Wav2Vec2Processor
#
#
# class ObscenityWordsRecognizer(SpeechRecognitionModel):
#
#     def __init__(self, model_path: str):
#         super().__init__(model_path)
#
#         self.__mode = 'silence'
#
#     def mute_word(self, audio_path, mode):
#         resampled_audio = self.resample_audio(audio_path, 16000)
#         transcribe = self.transcribe([resampled_audio])
#         y, sr = librosa.load(resampled_audio)
#         # words_coords = find_words(transcribe)
#         # result = cover_by_sound(y, words_coords, mode)
#         sf.write("result_" + audio_path, y, sr, format='wav', subtype='PCM_16')
#         return "result_" + audio_path
#
#     @staticmethod
#     def resample_audio(audio_path: str, target_rate: int) -> str:
#         y, sr = librosa.load(audio_path)
#         y = librosa.to_mono(y)
#         librosa.resample(y, sr, target_rate)
#         sf.write("RS_" + audio_path, y, target_rate, format='wav', subtype='PCM_16')
#         return "RS_" + audio_path
#
#
if __name__ == "__main__":
    detector = SpeechRecognitionModel("./data/model_settings")
    # detector.model.save_pretrained('./data/model_settings')
    # detector.processor.save_pretrained('./data/processor_settings')
#     # detector.processor = Wav2Vec2Processor.from_pretrained("/content/gdrive/MyDrive/wav2vec2-large-xlsr-russian-demo")
#     # detector.model.save_pretrained('/content/gdrive/MyDrive/wav2vec2-large-xlsr-russian-demo')
