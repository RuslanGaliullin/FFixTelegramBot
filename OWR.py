import json
import os
from huggingsound import SpeechRecognitionModel
import librosa
import soundfile as sf
from transformers import AutoModelForCTC, Wav2Vec2Processor
from fonetika.soundex import RussianSoundex
from fonetika.distance import PhoneticsInnerLanguageDistance
import numpy as np
from playsound import playsound


class ObscenityWordsRecognizer(SpeechRecognitionModel):

    def __init__(self):
        super().__init__("rmgaliullin/wav2vec2-based-obscenity-detector", letter_case='lowercase')
        self.__phonetic_word_codes = self.__get_phonetic_for_words("data/obscenity_words.json")
        # processor = Wav2Vec2Processor.from_pretrained("rmgaliullin/wav2vec2-large-xlsr-53-demo-colab")

    def mute_words(self, audio_path: str, mode):
        location = os.path.dirname(os.path.realpath(__file__))
        resampled_audio = self.resample_audio(audio_path, 16000)
        transcribe = self.transcribe([resampled_audio])
        y, sr = librosa.load(resampled_audio, mono=True, sr=16000)
        words_to_delete = self.__find_words(transcribe)
        self.__cover_by_sound(y, words_to_delete, mode)
        print(len(y), sr)
        directory = os.path.dirname(audio_path)
        full_name = os.path.basename(audio_path)
        sf.write(os.path.join(directory, "result_" + full_name), y, sr, format='wav', subtype="PCM_16")

        # y, sr = librosa.load(os.path.join(directory, "result_" + full_name), mono=True, sr=16000)

        return os.path.join(directory, "result_" + full_name)

    @staticmethod
    def resample_audio(audio_path: str, target_rate: int) -> str:
        directory = os.path.dirname(audio_path)
        full_name = os.path.basename(audio_path)
        # TODO
        y, sr = librosa.load(audio_path)
        # print(audio_path, sr)
        y = librosa.to_mono(y)
        new_samples = librosa.resample(y, sr, target_rate)
        sf.write(os.path.join(directory, "RS_" + full_name), new_samples, target_rate, format='wav', subtype="PCM_16")
        return os.path.join(directory, "RS_" + full_name)

    # TODO
    def __find_words(self, transcribe_result: list[dir]) -> list[tuple]:
        soundex = RussianSoundex(delete_first_letter=True)
        result = []
        sentence = transcribe_result[0]["transcription"].split()
        current_word_index = 0
        print(sentence)
        for index, value in enumerate(transcribe_result[0]['transcription'] + ' '):
            if value == ' ':
                if soundex.transform(sentence[current_word_index]) in self.__phonetic_word_codes.keys():
                    print(transcribe_result[0]["transcription"][index - len(sentence[current_word_index])])
                    print(transcribe_result[0]["transcription"][index - 1])
                    print(transcribe_result[0]["end_timestamps"][-1] * 16)
                    result.append(
                        (transcribe_result[0]["start_timestamps"][index - len(sentence[current_word_index])] * 16,
                         transcribe_result[0]["end_timestamps"][index - 1] * 16))
                current_word_index += 1
        print(result)
        return result

    @staticmethod
    def __get_phonetic_for_words(path_to_words: str) -> dict[str, str]:
        soundex = RussianSoundex(delete_first_letter=True)
        result = {}
        with open(path_to_words, 'rb') as fp:
            words = json.load(fp)
        for i in words:
            result[soundex.transform(i)] = i
        return result

    def __cover_by_sound(self, samples, words_to_delete, mode):
        for i in words_to_delete:
            samples[i[0]:i[1]] = np.zeros(i[1] - i[0])


if __name__ == "__main__":
    detector = ObscenityWordsRecognizer()
    a = detector.mute_words("/Users/ruslangaliullin/FFixTelegramBot/data/audio/awesome_test.wav", "silence")
    print(a)
    playsound("/Users/ruslangaliullin/FFixTelegramBot/data/audio/ressult_awesome_test.wav")
# detector = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
# detector.model.save_pretrained('./data/model_settings')
# detector.processor.save_pretrained('./data/processor_settings')
# detector.processor = Wav2Vec2Processor.from_pretrained("rmgaliullin/wav2vec2-large-xlsr-53-demo-colab")
#     # detector.model.save_pretrained('/content/gdrive/MyDrive/wav2vec2-large-xlsr-russian-demo')
