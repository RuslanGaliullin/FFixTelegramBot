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
import inspect
from huggingsound.metrics import cer, wer

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
    def __get_phonetic_for_words(path_to_words: str) -> dict[str, str]:
        soundex = RussianSoundex(delete_first_letter=True)
        result_word_to_code = {}
        with open(path_to_words, 'rb') as fp:
            words = json.load(fp)
        for i in words:
            result_word_to_code[i] = soundex.transform(i)
        return result_word_to_code

    @staticmethod
    def __cover_by_sound(samples: np.ndarray[np.float32], words_to_delete: list[tuple], mode: str) -> None:
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

    def __init__(self):
        self.ASR = SpeechRecognitionModel("rmgaliullin/wav2vec2-based-obscenity-detector", letter_case='lowercase')
        self.__phonetic_word_codes = self.__get_phonetic_for_words(os.path.join(
            os.path.split(inspect.getfile(self.__class__))[0], 'data', 'obscenity_words.json'))
        logger.info("Initialization of the model has been completed")

    def mute_words(self, audio_path: str, mode: str):
        audio_file = sf.SoundFile(audio_path)
        origin_samples = audio_file.read()
        origin_sr = audio_file.samplerate
        logger.info(
            f"Audio is loaded with {origin_sr} sample rate, {audio_file.channels} channels, type {audio_file.subtype}")
        resampled_audio = self.resample_audio(audio_path, 16000)

        # y, sr = librosa.load(resampled_audio, mono=True, sr=16000)

        words_to_delete = self.__find_words(self.ASR.transcribe([resampled_audio])[0], origin_sr)
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
        new_samples = librosa.resample(y, orig_sr=sr, target_sr=target_rate)
        sf.write(os.path.join(directory, "RS_" + full_name), new_samples, target_rate, format='wav', subtype="PCM_16")
        return os.path.join(directory, "RS_" + full_name)

    def __ow_probability(self, word: str, probabilities: list[float]) -> (float, int):
        result = 0
        res_ow = ''
        word_t = ObscenityWordsRecognizer.soundex.transform(word)
        s_len = min(int((len(word_t) + 1) / 2), len(word_t))
        for ow in self.__phonetic_word_codes.keys():
            ow_t = ObscenityWordsRecognizer.soundex.transform(ow)
            ow_s = min(len(ow_t), s_len)
            if word_t[0:ow_s] == ow_t[0:ow_s]:
                indexes = pylcs.lcs_sequence_idx(word, ow)
                probability = 0
                for index, value in enumerate(indexes):
                    if value != -1:
                        probability += probabilities[index]
                dist = Levenshtein.distance(ow, word)
                probability = probability / max(len(word), len(ow))
                if (probability / (dist + 1) ** 0.25) ** dist >= result:
                    result = (probability / (dist + 1) ** 0.25) ** dist
                    res_ow = ow
        return int(result * 10) / 10, res_ow

    def __find_words(self, transcribe_result: dir, sample_rate: int, evaluation: list[str] = None) -> list[tuple]:
        o_words = []
        sentence = transcribe_result['transcription']
        logger.info(f"Transcription: {sentence}")
        probabilities = transcribe_result['probabilities']
        start_timestamps = transcribe_result["start_timestamps"]
        end_timestamps = transcribe_result["end_timestamps"]

        for length in range(2, 8):
            for i in range(0, len(sentence) - length + 1):
                probability = self.__ow_probability(sentence[i:(i + length)], probabilities[i:(i + length)])
                if probability[0] > 0.5:
                    o_words.append((probability[0], length, i, probability[1]))
        o_words.sort()
        ow_timestamps = []
        recognized_ow_indexes = []
        ows = []
        while len(o_words) != 0:
            word = o_words.pop()
            p = word[0]
            word = (word[2], word[2] + word[1] - 1, word[3])
            is_valid = True
            for w in recognized_ow_indexes:
                if not ((word[0] < w[0] and word[1] < w[0]) or (w[0] < word[0] and w[1] < word[0])):
                    is_valid = False
                    break
            if is_valid:
                logger.info(
                    f"Detected word: {sentence[word[0]:(word[1] + 1)]};  Probability: {p}; Real word: {word[2]}")
                ows.append(word[2])
                recognized_ow_indexes.append(word)
                length = word[1] - word[0] + 1
                start = start_timestamps[word[0]] * (sample_rate // 1000)
                end = end_timestamps[word[1]] * (sample_rate // 1000)
                letter_len = (end - start) // length
                start -= 2 * letter_len
                end += 2 * letter_len
                if start < 0:
                    start = 0
                if end > end_timestamps[-1] * (sample_rate // 1000):
                    end = end_timestamps[-1] * (sample_rate // 1000)
                ow_timestamps.append((start, end))
        # ows - all recognised words
        if evaluation is not None:
            evaluation.extend(ows)
        return ow_timestamps

    def evaluate(self, audio_transcript_data) -> dict:
        data = []
        true_positive = 0
        total = 0
        for test in audio_transcript_data:
            prediction_ow = []
            text = open(test["transcript_path"]).read().strip()
            data.append({'path': test["audio_path"], 'transcription': text})
            reference_ow = [i for i in text.split() if i in self.__phonetic_word_codes.keys()]
            reference_ow.sort()
            self.__find_words(self.ASR.transcribe([test["audio_path"]])[0], 16000, prediction_ow)
            prediction_ow.sort()
            true_positive += len(prediction_ow)
            total += len(reference_ow)
        if total == 0:
            total = 1
            true_positive = 0
        return {"total cer": self.ASR.evaluate(data)["cer"],
                "total wer": self.ASR.evaluate(data)["wer"],
                "rate of word detection": true_positive / total}


if __name__ == "__main__":
    detector = ObscenityWordsRecognizer()
    evaluation = detector.evaluate(
        [{'audio_path': os.path.join("test", "audio", f'audio_{i}.wav'),
          'transcript_path': os.path.join("test", "transcript", f'transcript_{i}.txt')}
         for i in range(6)
         ])
    with open(os.path.join("test", 'result.json'), 'w') as outfile:
        outfile.write(json.dumps(evaluation))

