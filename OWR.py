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
    def __get_phonetic_for_words(path_to_words: str) -> dict[str, str]:
        soundex = RussianSoundex(delete_first_letter=True)
        result = {}
        with open(path_to_words, 'rb') as fp:
            words = json.load(fp)
        for i in words:
            result[soundex.transform(i)] = i
        return result

    def __cover_by_sound(self, samples, words_to_delete, mode):
        if mode == 's':
            for i in words_to_delete:
                samples[i[0]:i[1]] = np.zeros(i[1] - i[0])
        if mode == 'b':
            for i in words_to_delete:
                samples[i[0]:i[1]] = librosa.tone(800, i[1] - i[0], sr=16000)

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
        word = RussianSoundex(delete_first_letter=True).transform(word)
        for ow in self.__phonetic_word_codes.keys():
            indexes = pylcs.lcs_sequence_idx(ow, word)
            counter = 0
            probability = 0
            for index, value in enumerate(indexes):
                if (value == -1):
                    probability += probabilities[index]
                    counter += 1
                else:
                    probability += 10 * probabilities[index]
                    counter += 10
            probability = (probability / counter) ** Levenshtein.distance(ow, word)
            result = max(probability, result)
        return int(result * 10) / 10

    # TODO
    def __find_words(self, transcribe_result: list[dir]) -> list[tuple]:
        o_words = []
        sentence = transcribe_result[0]['transcription']
        probabilities = [value for index, value in enumerate(transcribe_result[0]['probabilities']) if
                         sentence[index] != ' ']
        start_timestamps = [value for index, value in enumerate(transcribe_result[0]["start_timestamps"]) if
                            sentence[index] != ' ']
        end_timestamps = [value for index, value in enumerate(transcribe_result[0]["end_timestamps"]) if
                          sentence[index] != ' ']
        sentence = str([value for index, value in enumerate(sentence) if
                        sentence[index] != ' '])

        for length in range(2, 8):
            for i in range(0, len(sentence) - length):
                probability = self.__ow_probability(sentence[i:(i + length)], probabilities[i:(i + length)])
                if (probability >= 0.5):
                    o_words.append((probability, length, i))
        o_words.sort(reverse=True)
        ow_timestamps = []
        recongnized_ow_indexes = []
        while len(o_words) != 0:
            word = o_words.pop()
            word = (word[1], word[1] + word[2] - 1)
            is_valid = True
            for w in recongnized_ow_indexes:
                if not ((word[0] < w[0] and word[1] < w[0]) or (w[0] < word[0] and w[1] < word[0])):
                    is_valid = False
            if (is_valid):
                recongnized_ow_indexes.append(word)
                ow_timestamps.append((start_timestamps[word[0]] * 16, end_timestamps[word[1]] * 16))
        print(ow_timestamps)
        return ow_timestamps


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
