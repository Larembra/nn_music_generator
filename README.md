Этот проект - рекуррентная нейронная сеть для генерации музыки. Она обучалась на/генерирует текстовое представление музыки в abc-нотации.

Пример записи в abc-нотации:
X:24
T:Jash of Mount Driny
Z: id:dc-ocarolan-20
M:C
L:1/8
K:D Major
AG|F2Ad fdAF|G2Bd gdBG|F2Ad fdAF|E2Ac ecAG|!
F2Ad fdAF|G2Bd gbag|fafd egec|d2f2 d2:|!
fg|a2fd Adfa|g2ec Aceg|a2fd Adgf|(3efe (3dcB A2fg|!
a2fd Adfa|gfef gbaf|gfef gbag|fgec d2:|!

Для работы с моделью скачай все с ветви master и запусти файл "nn_for_music.py".

Для того чтобы прослушать музыку, нужен проигрыватель abc-нотации, например https://abc.rectanglered.com/ (вставь туда текст музыки).

Также в проекте есть файл с примерами сгенерированных музыкальных произведений "generated_song_examples.abc".
