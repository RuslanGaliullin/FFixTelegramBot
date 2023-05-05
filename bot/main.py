import os

import telebot
from telebot import types
from pydub import AudioSegment
from OWR import ObscenityWordsRecognizer
import soundfile

bot = telebot.TeleBot("6189522732:AAE6lLTknaQcz008aDRykaiFrkwnE1OJAN8", parse_mode=None)
split_string = 'SL96illlSfCB5zP'
files = []
if not os.path.isdir('recieved_files'):
    os.mkdir('recieved_files')
else:
    for file in os.listdir('recieved_files'):
        pass  # UNDO
        # os.remove(os.path.join('recieved_files', file))
detector = ObscenityWordsRecognizer()


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Добро пожаловать! \nДанный бот создан для того, чтобы упростить "
                                      "удаление нецензурных слов из аудиофайла путём их запикивания или приглушения.\n\n"
                                      "Для обработки файла просто отправьте его личным сообщением.\n\n"
                                      "Требования к обрабатываемому аудиофайлу:\n"
                                      "- Расширение: .wav или .mp3\n"
                                      "- Максимальный размер: 20 Мб\n"
                                      "- Минимальная частота дискретизации: 16 кГц\n"
                                      "- Минимальная глубина кодирования: 16 бит\n\n")


@bot.message_handler(commands=['help'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Требования к обрабатываемому аудиофайлу:\n"
                                      "- Расширение .wav или .mp3\n"
                                      "- Максимальный размер: 100 Мб\n"
                                      "- Минимальная частота дискретизации: 16 кГц\n"
                                      "- Минимальная глубина кодирования: 16 бит\n\n"
                                      "Создатели бота:\n<a href='https://t.me/vladvslv'>Васильев Владислав</a>\n"
                                      "<a href='https://t.me/R_Galiullin'>Галиуллин Руслан</a>", parse_mode="HTML")


@bot.message_handler(content_types=['voice'])
def voice_message(message):
    file_info = bot.get_file(message.voice.file_id)
    path_to_file = 'recieved_files/' + str(file_info.file_unique_id) + split_string + 'voice.ogg'
    downloaded_file = bot.download_file(file_info.file_path)
    with open(path_to_file, 'wb') as f:
        f.write(downloaded_file)

    data, sr = soundfile.read(path_to_file)
    soundfile.write(path_to_file.split('.')[0] + '.wav', data, sr)
    # AudioSegment.from_ogg(path_to_file).export(path_to_file.split('.')[0] + '.wav', format='wav')
    # os.remove(path_to_file)
    path_to_file = path_to_file.split('.')[0] + '.wav'
    extension = 'v'

    files.append(path_to_file)
    beep = types.InlineKeyboardButton('Запикать', callback_data='b' + extension + str(len(files) - 1))
    silence = types.InlineKeyboardButton('Приглушить', callback_data='s' + extension + str(len(files) - 1))
    cancel = types.InlineKeyboardButton('Отменить обработку', callback_data='c' + extension + str(len(files) - 1))
    buttons = types.InlineKeyboardMarkup([[beep, silence], [cancel]])
    bot.send_message(message.chat.id, 'Выберите необходимое действие', reply_markup=buttons)


@bot.message_handler(content_types=['audio'])
def message_reply(message):
    file_info = bot.get_file(message.audio.file_id)
    path_to_file = 'recieved_files/' + str(file_info.file_unique_id) + split_string + message.audio.file_name
    if not path_to_file.split('.')[1] in ['wav', 'mp3']:
        bot.send_message(message.chat.id, 'Неверный формат файла.')
        return
    downloaded_file = bot.download_file(file_info.file_path)
    with open(path_to_file, 'wb') as f:
        f.write(downloaded_file)
    extension = 'w'
    if path_to_file.split('.')[1] == 'mp3':
        AudioSegment.from_mp3(path_to_file).export(path_to_file.split('.')[0] + '.wav', format='wav')
        # os.remove(path_to_file)
        path_to_file = path_to_file.split('.')[0] + '.wav'
        extension = 'm'
    files.append(path_to_file)
    beep = types.InlineKeyboardButton('Запикать', callback_data='b' + extension + str(len(files) - 1))
    silence = types.InlineKeyboardButton('Приглушить', callback_data='s' + extension + str(len(files) - 1))
    cancel = types.InlineKeyboardButton('Отменить обработку', callback_data='c' + extension + str(len(files) - 1))
    buttons = types.InlineKeyboardMarkup([[beep, silence], [cancel]])
    bot.send_message(message.chat.id, 'Выберите необходимое действие', reply_markup=buttons)


@bot.callback_query_handler(func=lambda c: c.data[0] == "s" or c.data[0] == "b")
def answer_callback_query(callback_query: types.CallbackQuery):
    bot.answer_callback_query(callback_query.id)
    bot.delete_message(chat_id=callback_query.message.chat.id,
                       message_id=callback_query.message.message_id)
    if len(files) > int(callback_query.data[2:]) and os.path.exists(files[int(callback_query.data[2:])]):
        bot.send_message(callback_query.from_user.id, 'Выполняется обработка файла')
        file_path = files[int(callback_query.data[2:])]
        with open("test_files.txt", 'a') as file:
            file.write(file_path + '\n')
        result_file_path = detector.mute_words(file_path, callback_query.data[0])
        # os.remove(file_path)
        file_path = result_file_path
        if callback_query.data[1] == 'm':
            AudioSegment.from_wav(file_path).export(file_path.split('.')[0] + '.mp3', format='mp3')
            # os.remove(file_path)
            file_path = file_path.split('.')[0] + '.mp3'
        # os.rename(file_path, os.path.join('recieved_files', file_path.split(split_string)[1]))
        # file_path = os.path.join('recieved_files', file_path.split(split_string)[1])
        file = open(file_path, 'rb')
        if callback_query.data[1] == 'v':
            bot.send_voice(callback_query.from_user.id, file)
        else:
            bot.send_audio(callback_query.from_user.id, file)
        file.close()
        # os.remove(file_path)
    else:
        bot.send_message(callback_query.message.chat.id, 'Файл удалён из системы')


@bot.callback_query_handler(func=lambda c: c.data[0] == "c")
def answer_callback_query(callback_query: types.CallbackQuery):
    bot.answer_callback_query(callback_query.id)
    bot.delete_message(chat_id=callback_query.message.chat.id,
                       message_id=callback_query.message.message_id)
    if len(files) > int(callback_query.data[2:]) and os.path.exists(files[int(callback_query.data[2:])]):
        file_path = files[int(callback_query.data[2:])]
        bot.send_message(callback_query.from_user.id, 'Обработка отменена')
        # os.remove(file_path)
    else:
        bot.send_message(callback_query.message.chat.id, 'Файл удалён из системы')


if __name__ == "__main__":
    bot.infinity_polling(none_stop=True)
