import os
import telebot
from telebot import types

bot = telebot.TeleBot("6189522732:AAE6lLTknaQcz008aDRykaiFrkwnE1OJAN8", parse_mode=None)
split_string = 'SL96illlSfCB5zP'
files = []


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Добро пожаловать! \nДанный бот создан для того, чтобы упростить "
                                      "удаление цецензурных слов из аудиофайла путём их запикивания или приглушения.\n\n"
                                      "Для обработки файла просто отправьте его личным сообщением.\n\n"
                                      "Требования к обрабатываемому аудиофайлу:\n"
                                      "- Расширение: .wav или .mp3\n"
                                      "- Максимальный размер: 100 Мб\n"
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
                                      "<a href='https://t.me/R_Galiullin'>Галлиулин Руслан</a>", parse_mode="HTML")


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
    files.append(path_to_file)
    beep = types.InlineKeyboardButton('Запикать', callback_data='b' + str(len(files) - 1))
    silence = types.InlineKeyboardButton('Приглушить', callback_data='s' + str(len(files) - 1))
    cancel = types.InlineKeyboardButton('Отменить обработку', callback_data='c' + str(len(files) - 1))
    buttons = types.InlineKeyboardMarkup([[beep, silence], [cancel]])
    bot.send_message(message.chat.id, 'Выберите необходимое действие', reply_markup=buttons)


@bot.callback_query_handler(func=lambda c: c.data[0] == "s" or c.data[0] == "b")
def answer_callback_query(callback_query: types.CallbackQuery):
    bot.answer_callback_query(callback_query.id)
    bot.delete_message(chat_id=callback_query.message.chat.id,
                       message_id=callback_query.message.message_id)
    bot.send_message(callback_query.from_user.id, 'Выполняется обработка файла')
    file_path = files[int(callback_query.data[1:])]
    with open(file_path, 'rb') as file:
        f = file.read()

    # обработка файла

    bot.send_document(callback_query.from_user.id, f, visible_file_name=file_path.split(split_string)[1])
    os.remove(file_path)


@bot.callback_query_handler(func=lambda c: c.data[0] == "c")
def answer_callback_query(callback_query: types.CallbackQuery):
    bot.answer_callback_query(callback_query.id)
    file_path = files[int(callback_query.data[1:])]
    bot.delete_message(chat_id=callback_query.message.chat.id,
                       message_id=callback_query.message.message_id)
    bot.send_message(callback_query.from_user.id, 'Обработка отменена')
    os.remove(file_path)


bot.infinity_polling(none_stop=True)
