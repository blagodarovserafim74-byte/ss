# cucumber-detection1

Проект для обучения модели, которая находит огурцы на фото/видео и обводит их прямоугольником.
Базируется на YOLO (Ultralytics) с возможностью обучения на GPU.

## Структура

```
cucumber-detection1/
  configs/
    dataset.yaml          # описание датасета для YOLO
    cucumber_prior.yaml   # «начальные знания» о форме/цвете
  data/
    (сюда кладите датасет)
  models/
  scripts/
  src/
    bootstrap_labels.py   # псевдо-разметка по цвету/форме
    auto_train.py         # авто-обучение: разметка + тренировка
    config.py             # загрузка конфигов
    heuristics.py         # эвристики формы/цвета огурцов
    infer_image.py        # инференс на изображении
    infer_video.py        # инференс на видео/камере
    train.py              # обучение модели
  run.py                  # запуск проекта через меню
  weights/
```

## Быстрый старт

1. Установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Подготовьте датасет в YOLO формате и укажите путь в `configs/dataset.yaml`.

3. (Опционально) Сгенерируйте псевдо-разметку с помощью базовых знаний об огурцах:

```bash
python -m src.bootstrap_labels \
  --images ./data/images \
  --labels ./data/labels
```

4. Запустите обучение (GPU, если доступна):

```bash
python -m src.train \
  --dataset configs/dataset.yaml \
  --epochs 50 \
  --imgsz 640
```

5. Инференс на фото:

```bash
python -m src.infer_image \
  --weights weights/best.pt \
  --image ./data/test.jpg \
  --output ./data/output.jpg
```

6. Инференс на видео или IP-камере:

```bash
python -m src.infer_video \
  --weights weights/best.pt \
  --source ./data/test.mp4
```

Для IP-камеры используйте `--source rtsp://user:pass@ip:port/stream`.

## Запуск через меню

Все основные действия можно запускать через единый файл:

```bash
python run.py
```

Меню позволяет:
- запустить автообучение (псевдо-разметка + обучение);
- выполнить отдельное обучение с сохранением чекпоинтов;
- сделать псевдо-разметку;
- выполнить инференс на фото/видео/камере.

После обучения веса копируются в папку `weights/` с именем, соответствующим эксперименту.

## «Начальные знания» об огурцах

В `configs/cucumber_prior.yaml` зафиксированы базовые признаки:
- Цвет: зелёный диапазон в HSV.
- Форма: вытянутый объект (отношение сторон, площадь, заполненность).

Эвристики используются в `src/heuristics.py` и позволяют получить первичную
псевдо-разметку для старта обучения, особенно если данных мало.
