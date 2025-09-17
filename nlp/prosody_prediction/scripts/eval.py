from nlp.prosody_prediction.eval_interface import ProsodyPredictionInterface

if __name__ == "__main__":
    ckpt_path = "M:\\Ilya\\JustAI\\07_Apr_2025_16_43_34_prosody_predictor_epoch=20_step=131250_category_EER=0.3664.ckpt"
    device = "cpu"

    text = """
    Главная особенность этой технологии — создание замкнутого цикла обучения, где ИИ сам выступает и учеником, и учителем.
    Система работает по принципу внутренней обратной связи: одна часть модели генерирует ответы, а другая выступает «судьей»,
    оценивая их качество и соответствие заданным критериям.
    Если ответ удовлетворяет требованиям, модель получает «вознаграждение» и запоминает успешную стратегию.
    """

    interface = ProsodyPredictionInterface(ckpt_path=ckpt_path, lang="RU", device=device)
    text_with_prosody = interface.predict(text)

    for sent_id, sent in enumerate(text_with_prosody.sents):
        print(f"Sentence {sent_id}:")
        for token in sent.tokens:
            print(f"\t{token.text} -- {token.prosody}")
        print("***************************\n")
