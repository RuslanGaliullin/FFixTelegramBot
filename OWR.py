import json
import os
import sys

from huggingsound import SpeechRecognitionModel
import librosa
import Levenshtein
import soundfile as sf
from fonetika.soundex import RussianSoundex
import numpy as np
from playsound import playsound
import pylcs
import logging

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


class ObscenityWordsRecognizer:
    @staticmethod
    def __compare_phonemes(word1: str, word2: str, parts: int) -> bool:
        soundex = RussianSoundex(delete_first_letter=True, reduce_word=True)
        return soundex.transform(word1[:len(word1) // parts]) == soundex.transform(word2[:len(word2) // parts])

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

    @staticmethod
    def __cover_by_sound(samples: np.ndarray[np.float32], words_to_delete: list[tuple], mode: str) -> None:
        if mode == 's':
            for i in words_to_delete:
                for channel in range(samples.ndim):
                    samples[i[0]:i[1], channel] = np.zeros(i[1] - i[0])
        if mode == 'b':
            for i in words_to_delete:
                for channel in range(samples.ndim):
                    samples[i[0]:i[1], channel] = librosa.tone(800, length=i[1] - i[0], sr=16000)

    def __init__(self):
        self.ASR = SpeechRecognitionModel("rmgaliullin/wav2vec2-based-obscenity-detector", letter_case='lowercase')
        self.__phonetic_word_codes = self.__get_phonetic_for_words("data/obscenity_words.json")

    def mute_words(self, audio_path: str, mode: str):
        audio_file = sf.SoundFile(audio_path)
        origin_samples = audio_file.read()
        origin_sr = audio_file.samplerate
        logger.info(
            f"Audio is loaded with {origin_sr} sample rate, {audio_file.channels} channels, type {audio_file.subtype}")
        resampled_audio = self.resample_audio(audio_path, 16000)

        # y, sr = librosa.load(resampled_audio, mono=True, sr=16000)

        words_to_delete = self.__find_words(self.ASR.transcribe([resampled_audio]), origin_sr)
        self.__cover_by_sound(origin_samples, words_to_delete, mode)

        directory = os.path.dirname(audio_path)
        full_name = os.path.basename(audio_path)

        sf.write(os.path.join(directory, "result_" + full_name), origin_samples, origin_sr, format='wav',
                 subtype=audio_file.subtype)

        os.remove(resampled_audio)
        return os.path.join(directory, "result_" + full_name)

    @staticmethod
    def resample_audio(audio_path: str, target_rate: int) -> str:
        directory = os.path.dirname(audio_path)
        full_name = os.path.basename(audio_path)
        y, sr = librosa.load(audio_path)
        y = librosa.to_mono(y)
        new_samples = librosa.resample(y, sr, target_rate)
        sf.write(os.path.join(directory, "RS_" + full_name), new_samples, target_rate, format='wav', subtype="PCM_16")
        return os.path.join(directory, "RS_" + full_name)

    def __ow_probability(self, word: str, probabilities: list[float]) -> float:
        result = 0
        s_len = int(len(word) / 3)
        for ow in self.__phonetic_word_codes[1].keys():
            if word[0:s_len] == ow[0:s_len]:
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
    def __find_words(self, transcribe_result: list[dir], sample_rate: int) -> list[tuple]:
        o_words = []
        sentence = transcribe_result[0]['transcription']
        logger.info(f"Transcription: {sentence}")
        probabilities = transcribe_result[0]['probabilities']
        start_timestamps = transcribe_result[0]["start_timestamps"]
        end_timestamps = transcribe_result[0]["end_timestamps"]

        for length in range(2, 8):
            for i in range(0, len(sentence) - length + 1):
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
                logger.info(f"Detected word: {sentence[word[0]:(word[1] + 1)]}  Probability: {p}")
                recognized_ow_indexes.append(word)
                ow_timestamps.append(
                    (start_timestamps[word[0]] * (sample_rate // 1000) - (sample_rate // 1000),
                     end_timestamps[word[1]] * (sample_rate // 1000) + (sample_rate // 1000)))
        return ow_timestamps


if __name__ == "__main__":
    detector = ObscenityWordsRecognizer()
    dir_path = os.path.join(os.getcwd(), "data", "audio")
    file_name = "test_sample_44_1khz_stereo.wav"
    a = detector.mute_words(os.path.join(dir_path, file_name), "s")
    print(a)
    playsound(os.path.join(dir_path, "result_" + file_name))
