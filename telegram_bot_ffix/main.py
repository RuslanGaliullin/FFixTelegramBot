import telebot
from telebot import types

bot = telebot.TeleBot("6189522732:AAE6lLTknaQcz008aDRykaiFrkwnE1OJAN8", parse_mode=None)

instruction = ""
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


@bot.message_handler(content_types=["audio"])
def message_reply(message):
    beep = types.InlineKeyboardButton('Запикать',callback_data="b")
    silence = types.InlineKeyboardButton('Приглушить', callback_data="s")
    cancel = types.InlineKeyboardButton('Отменить обработку', callback_data="c")
    buttons=types.InlineKeyboardMarkup([[beep,silence],[cancel]])

    bot.send_message(message.chat.id, 'Выберите необходимое действие', reply_markup=buttons)


@bot.callback_query_handler(func=lambda c: c.data == "s" or c.data == "b")
def answer_callback_query(callback_query: types.CallbackQuery):
    bot.answer_callback_query(callback_query.id)
    bot.send_message(callback_query.from_user.id, 'Выполняется обработка файла')

@bot.callback_query_handler(func=lambda c: c.data =="c")
def answer_callback_query(callback_query: types.CallbackQuery):
    bot.answer_callback_query(callback_query.id)
    bot.send_message(callback_query.from_user.id, 'Обработка отменена')

bot.infinity_polling(none_stop=True)
