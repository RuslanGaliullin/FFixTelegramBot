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
    soundex = RussianSoundex(delete_first_letter=True, reduce_word=True)

    @staticmethod
    def __get_phonetic_for_words(path_to_words: str) -> tuple[dict[str, str], dict[str, str]]:
        """
        Сохранение фонограм для всех слов из базы нецензурных слов.

        Parameters:
        ----------
            path_to_words: str
                Json-файл со словарем нецензурных слов
        Returns:
        ----------
            tuple[dict[str, str], dict[str, str]]:
                Пара из словарей типа код - слово, слово - код для всех нецензурных слов
        """

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
        """
        Перекрытие семплов звуком в указанных подотрезках.

        Parameters:
        ----------
            samples: np.ndarray[np.float32]
                Изначальный массив семплов
            words_to_delete: list[tuple]
                Список пар вида (начало, конец) с индексами подотрезков для наложения звука
            mode: str
                Тип изменения: 's' - тишина, 'b' - характерный звук 'бип'
        """

        if mode == 's':
            for i in words_to_delete:
                if samples.ndim == 1:
                    samples[i[0]:i[1]] = np.zeros(i[1] - i[0])
                else:
                    samples[i[0]:i[1]] = np.zeros((i[1] - i[0], samples.ndim))
        if mode == 'b':
            for i in words_to_delete:
                if samples.ndim == 1:
                    samples[i[0]:i[1]] = librosa.tone(600, length=i[1] - i[0], sr=16000)
                else:
                    samples[i[0]:i[1]] = np.tile((librosa.tone(600, length=i[1] - i[0], sr=16000)),
                                                 (samples.ndim, 1)).transpose()

    """
        Obscenity Words Recognizer.
        
        Инициализация предобученных весов из rmgaliullin/wav2vec2-based-obscenity-detector модели
        Сохранение кодов произношения ругательных слов из словаря data/obscenity_words.json
    """

    def __init__(self):
        self.ASR = SpeechRecognitionModel("rmgaliullin/wav2vec2-based-obscenity-detector", letter_case='lowercase')
        self.__phonetic_word_codes = self.__get_phonetic_for_words("data/obscenity_words.json")

    def mute_words(self, audio_path: str, mode: str) -> str:
        """
        Наложение звука поверх нецензурных слов из файла.

        Parameters:
        ----------
            audio_path: str
                Путь до аудио-файла для поиска нецензурных слов

            mode: str
                Тип изменения: 's' - тишина, 'b' - характерный звук 'бип'

        Returns:
        ----------
            str:
                Путь до аудио-файла с перекрытыми нецензурными словами
        """

        audio_file = sf.SoundFile(audio_path)
        origin_samples = audio_file.read()
        origin_sr = audio_file.samplerate

        logger.info(
            f"Audio is loaded with {origin_sr} sample rate, {audio_file.channels} channels, type {audio_file.subtype}")
        resampled_audio = self.resample_audio(audio_path, 16000)

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
        """
        Приведение аудио-файла к типу с частотой дискретизации 16кГц в одноканальном режиме.

        Parameters:
        ----------
            audio_path: str
                Путь до аудио-файла для изменения параметров

            target_rate: int
                Целевая частота для изменения

        Returns:
        ----------
            str:
                Путь до аудио-файла с измененными параметрами каналов и частоты дискретизации
        """

        directory = os.path.dirname(audio_path)
        full_name = os.path.basename(audio_path)
        y, sr = librosa.load(audio_path)
        y = librosa.to_mono(y)
        new_samples = librosa.resample(y, orig_sr=sr, target_sr=target_rate)
        sf.write(os.path.join(directory, "RS_" + full_name), new_samples, target_rate, format='wav', subtype="PCM_16")
        return os.path.join(directory, "RS_" + full_name)

    def __ow_probability(self, word: str, probabilities: list[float]) -> float:
        """
        Вычисление вероятности принадлежности слова к классу нецензурных.

        Parameters:
        ----------
            word: str
                Проверяемое слово

            probabilities: float
                Степень уверенности в каждой букве в word

        Returns:
        ----------
            float:
                Итоговая вероятность того, что слова является нецензурным
        """

        result = 0
        word_t = ObscenityWordsRecognizer.soundex.transform(word)
        s_len = int(len(word_t) / 3)

        for ow in self.__phonetic_word_codes[1].keys():
            # Проверка, что первая треть слова звучит как начало какого-то нецензурного слова
            if word_t[0:s_len] == ObscenityWordsRecognizer.soundex.transform(ow)[0:s_len]:
                indexes = pylcs.lcs_sequence_idx(word, ow)
                probability = 0
                for index, value in enumerate(indexes):
                    if value != -1:
                        probability += probabilities[index]
                # Расстояние Левенштайна
                dist = Levenshtein.distance(ow, word)
                probability = probability / max(len(word), len(ow))
                result = max((probability / (dist + 1) ** 0.25) ** dist, result)
        return int(result * 10) / 10

    def __find_words(self, transcribe_result: list[dir], sample_rate: int) -> list[tuple]:
        """
        Нахождение диапазонов в sample_rate исходного аудио, которые содержат нецензурные слова.

        Parameters:
        ----------
            transcribe_result: list[dir]
                Список с транскрипциами для каждого аудио-файла в виде:

                [{
                    "transcription": str,
                    "start_timestemps": list[int],
                    "end_timestemps": list[int],
                    "probabilities": list[float]
                }, ...]

            sample_rate: int
                Частота дискретизации исходного аудио-файла

        Returns:
        ----------
            list[tuple]:
                Список подотрезков, которые соответствуют  нецензурным в списке сэмплов
        """

        o_words = []
        sentence = transcribe_result[0]['transcription']

        logger.info(f"Transcription: {sentence}")

        probabilities = transcribe_result[0]['probabilities']
        start_timestamps = transcribe_result[0]["start_timestamps"]
        end_timestamps = transcribe_result[0]["end_timestamps"]

        # Вычисление вероятности того, что это нецензурное слово для каждой подстроки длиной от 2 до 8
        for length in range(2, 8):
            for i in range(0, len(sentence) - length + 1):
                probability = self.__ow_probability(sentence[i:(i + length)], probabilities[i:(i + length)])
                if probability >= 0.5:
                    o_words.append((probability, length, i))

        o_words.sort()
        ow_timestamps = []
        recognized_ow_indexes = []

        # Сохранение позиций
        while len(o_words) != 0:
            word = o_words.pop()
            p = word[0]
            word = (word[2], word[2] + word[1] - 1)
            is_valid = True
            for w in recognized_ow_indexes:
                if not ((word[0] < w[0] and word[1] < w[0]) or (w[0] < word[0] and w[1] < word[0])):
                    is_valid = False
            if is_valid:
                logger.info(f"Detected word: {sentence[word[0]:(word[1] + 1)]};  Probability: {p}")
                recognized_ow_indexes.append(word)
                ow_timestamps.append(
                    (start_timestamps[word[0]] * (sample_rate // 1000) - (sample_rate // 1000),
                     end_timestamps[word[1]] * (sample_rate // 1000) + (sample_rate // 1000)))
        return ow_timestamps


if __name__ == "__main__":
    detector = ObscenityWordsRecognizer()
    # Папка с аудио-файлом
    dir_path = os.path.join(os.getcwd(), "data", "audio")
    # Название аудио-файла
    file_name = "audio_02.wav"
    # Перекрытие нецензурных слов в аудио-файле
    a = detector.mute_words(os.path.join(dir_path, file_name), "s")
    print(a)
    # Проигрование итогового аудио
    playsound(os.path.join(dir_path, "result_" + file_name))
