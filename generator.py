import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("models/essays")

model = GPT2LMHeadModel.from_pretrained("models/essays")

model.cuda()

text = "Запрос: Напиши описание к отелю Intourist Hotel Kolomenskoe (Интурист Коломенское), который находится на Каширское шоссе, д.39Б в городе Москва имеет такие особенности, как Возможно проживание с животными, Wi-Fi в номере, Wi-Fi в отеле, Зарядка электромобиля, Сауна, Парковка, Фитнес, Ресторан, Настольный теннис, Тренажерный зал, Лифт, Бар, Диетическое меню, Спортивные трансляции, Крытая парковка, Снек-бар, Аренда авто , Завтрак с собой, UnionPay, Конференц-зал, Обслуживание в номере, Магазин, Туалет с поручнями, Пеший туризм, Заказ такси, Прачечная/Химчистка, Бизнес-центр, Миски для животных, Ускоренная регистрация заезда/выезда, Переговорная комната, Индивидуальная регистрация заезда/выезда, Счета и закрывающие документы, Камера хранения/комната для багажа, Банкомат, Экскурсионное бюро, Факс/ксерокс, Регистрация иностранных граждан, Продажа билетов на шоу и мероприятия, Видеонаблюдение, Газеты/журналы, Консьерж-сервис, Сейф на ресепшен, Места для курения, Чистка обуви, Датчик дыма, Аптечка, Огнетушитель, Охраняемая территория\nОтвет: "
inpt = tok.encode(text, return_tensors="pt")
out = model.generate(inpt.cuda(), max_length=500, repetition_penalty=5.0, do_sample=True, top_k=5, top_p=0.95, temperature=1)
print(tok.decode(out[0]))
