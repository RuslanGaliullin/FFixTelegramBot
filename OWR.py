import json
import os
from huggingsound import SpeechRecognitionModel
import librosa
import Levenshtein
import soundfile as sf
from transformers import AutoModelForCTC, Wav2Vec2Processor
from fonetika.soundex import RussianSoundex
from fonetika.distance import PhoneticsInnerLanguageDistance
import numpy as np
from playsound import playsound
import pylcs


class ObscenityWordsRecognizer(SpeechRecognitionModel):
    @staticmethod
    def __get_phonetic_for_words(path_to_words: str) -> tuple[dict[str, str], dict[str, str]]:
        soundex = RussianSoundex(delete_first_letter=True)
        result_word_to_code = {}
        result_code_to_word = {}
        with open(path_to_words, 'rb') as fp:
            words = json.load(fp)
        for i in words:
            result_code_to_word[soundex.transform(i)] = i
            result_word_to_code[i] = soundex.transform(i)
        return result_code_to_word, result_word_to_code

    def __cover_by_sound(self, samples, words_to_delete, mode):
        if mode == 's':
            for i in words_to_delete:
                samples[i[0]:i[1]] = np.zeros(i[1] - i[0])
        if mode == 'b':
            for i in words_to_delete:
                samples[i[0]:i[1]] = librosa.tone(800, length = i[1] - i[0], sr=16000)

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

    def __ow_probability(self, word: str, probabilities: list[float]) -> float:
        result = 0
        for ow in self.__phonetic_word_codes[1].keys():
            indexes = pylcs.lcs_sequence_idx(word, ow)
            probability = 0
            for index, value in enumerate(indexes):
                if value != -1:
                    probability += probabilities[index]
            dist = Levenshtein.distance(ow, word)
            probability = probability / max(len(word), len(ow))
            result = max((probability / (dist + 1) ** 0.25) ** dist, result)
        return int(result * 10) / 10

    # TODO
    def __find_words(self, transcribe_result: list[dir]) -> list[tuple]:
        o_words = []
        sentence = transcribe_result[0]['transcription']
        print(sentence)
        probabilities = transcribe_result[0]['probabilities']
        start_timestamps = transcribe_result[0]["start_timestamps"]
        end_timestamps = transcribe_result[0]["end_timestamps"]

        for length in range(2, 8):
            for i in range(0, len(sentence) - length):
                probability = self.__ow_probability(sentence[i:(i + length)], probabilities[i:(i + length)])
                if probability >= 0.5:
                    o_words.append((probability, length, i))
        o_words.sort()
        ow_timestamps = []
        recognized_ow_indexes = []
        while len(o_words) != 0:
            word = o_words.pop()
            p = word[0]
            word = (word[2], word[2] + word[1] - 1)
            is_valid = True
            for w in recognized_ow_indexes:
                if not ((word[0] < w[0] and word[1] < w[0]) or (w[0] < word[0] and w[1] < word[0])):
                    is_valid = False
            if is_valid:
                print(p)
                print(sentence[word[0]:(word[1] + 1)])
                recognized_ow_indexes.append(word)
                ow_timestamps.append((start_timestamps[word[0]] * 16, end_timestamps[word[1]] * 16))
        return ow_timestamps


if __name__ == "__main__":
    detector = ObscenityWordsRecognizer()
    a = detector.mute_words("/home/vladislav/Files/Git/FFixTelegramBot/data/audio/audio_36.wav", "b")
    print(a)
    playsound("/home/vladislav/Files/Git/FFixTelegramBot/data/audio/result_audio_36.wav")
# detector = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
# detector.model.save_pretrained('./data/model_settings')
# detector.processor.save_pretrained('./data/processor_settings')
# detector.processor = Wav2Vec2Processor.from_pretrained("rmgaliullin/wav2vec2-large-xlsr-53-demo-colab")
#     # detector.model.save_pretrained('/content/gdrive/MyDrive/wav2vec2-large-xlsr-russian-demo')
