# Генератор музыки

Проект обучает символьную LSTM-модель на мелодиях в ABC-нотации и сохраняет сгенерированные мелодии как `.abc` файлы.

## Структура

- `src/music_generator/` — CLI, конфигурация, работа с ABC-данными и модель.
- `data/` — локальный учебный набор `irish.abc`.
- `configs/` — настройки обучения и генерации.
- `artifacts/` — контрольные точки и результат генерации.
- `logs/` — журналы запуска.
- `.env.example` — шаблон локальных переменных окружения.

## Установка

Требуется Python 3.10 или новее. TensorFlow может потребовать отдельного варианта установки для вашей ОС/GPU.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
Copy-Item .env.example .env
```

## CLI

```powershell
music-generator check-data
music-generator train --iterations 3000
music-generator generate --songs 3 --characters 6000
```

Также можно запускать без установки команды: `python -m music_generator.cli --help` после `pip install -e .`.
Аргументы `--resume` и `--fresh` управляют продолжением обучения с последней контрольной точки.
В `MUSIC_*` переменных из `.env` можно переопределить пути и параметры модели. Сгенерированные файлы
появляются в `artifacts/generated/`, журналы — в `logs/`.

## Данные и безопасность

Датасет находится локально, CLI не скачивает и не исполняет входные файлы. Не помещайте секреты в
исходники; локальный `.env`, журналы, окружение Python и бинарные веса исключены из Git. Каталог
вывода можно задать флагом `--output`.

## Как прослушать мелодии

1. Откройте [abc.rectanglered.com](https://abc.rectanglered.com/).
2. Скопируйте ABC-нотацию мелодии и вставьте её в редактор на сайте.
3. Нажмите кнопку воспроизведения на странице.

## Пример сгенерированных мелодий

```abc
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



X:60
T:Croley's
Z: id:dc-hornpipe-14
M:C|
L:1/8
K:D Major
AG|FDFA dfaf|gfec dcBA|G2BG F2AF|EDEF GBAG|!
FDFA dfaf|gfec dcBA|dfaf bgec|d2dc d2:|!
(3ABc|d2fd c2ec|BABc dBAF|G2BG F2AF|EDEF GBAG|!
FDFA dfaf|gfec dcBA|dfaf bgec|d2dc d2:|!



X:32
T:Weaver's
Z: id:dc-slide-33
M:6/8
L:1/8
K:D Major
g|fed B2d|A2d F2B|A2F D2F|E3 E2g|!
fed B2d|A2d F2B|A2F EFE|D3 D2:|!
e|fdf a3|baf a3|fdf a2f|g3 f2e|!
[1 fdf a3|baf a3|fdf a2f|d3 d2:|!
[2 faf efe|ded BdB|ABA GFE|D3 D2|]!



X:24
T:Where's the Cat
Z: id:dc-polka-42
M:2/4
L:1/8
K:A Major
a2 a>f|ec a>f|ec a>f|ec BA|!
a2 a>f|ec a2|ce B>c|BA A2:|!
ce e2|df f2|ce e2|cB B2|!
ce e2|df f2|ce B>c|BA A2:|!



X:33
T:High Level
Z: id:dc-hornpipe-30
M:C|
L:1/8
K:D Major
fe|dAFA DAFA|d2ec dcBc|A2ce A2ce|fdec dcBA|!
dAFA DAFA|d2ec dcBA|gfgb afdf|eABc d2:|!
B=c|dgg2 bgg2|dgbg agfe|A2ce A2ce|fdec dcBA|!
dgg2 bgg2|dgbg agef|gfgb afdf|eABc d2:|!
```
