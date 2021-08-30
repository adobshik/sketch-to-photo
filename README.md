# sketch-to-photo

В этом репозитории вы найдете модель, основанную на нейронной сети, которая берет контурное изображение (эскиз или рисунок) кошки и создает его цветную фотореалистическую версию.

Пример набросанного в paint от руки контурного изображения и его синтезированная фотореалистическая версия. 
![Image alt](https://github.com/adobshik/sketch-to-photo/blob/main/example/1cat.png)
![Image alt](https://github.com/adobshik/sketch-to-photo/blob/main/example/1mask.png)

![Image alt](https://github.com/adobshik/sketch-to-photo/blob/main/example/2cat.png)
![Image alt](https://github.com/adobshik/sketch-to-photo/blob/main/example/2mask.png)

![Image alt](https://github.com/adobshik/sketch-to-photo/blob/main/example/3cat.png)
![Image alt](https://github.com/adobshik/sketch-to-photo/blob/main/example/3mask.png)

Пример работы генератора на тренировочных данных:


![Image alt](https://github.com/adobshik/sketch-to-photo/blob/main/example/0a1f3266-f6a8-4d26-9314-7e3fb1d492f6.png)
![Image alt](https://github.com/adobshik/sketch-to-photo/blob/main/example/8b182abd-8191-4fe2-b81f-89b8f00f6a7c.png)

# Процесс подготовки данных:
Фотографии кошек были взяты с [kaggle](https://www.kaggle.com/crawford/cat-dataset) (всего 9997 фотографий). 

Для получения эскизов из фотографий котов была использована преднатренированная сеть [Contour-Detection-Pytorch](https://github.com/captanlevi/Contour-Detection-Pytorch) (воспроизведение статьи [Object Contour Detection with a Fully Convolutional Encoder-Decoder Network](https://arxiv.org/pdf/1603.04530.pdf)). Используемая предобученная модель для получения контуров и jupyter notebook с кодом подготовки данных доступны [здесь](https://drive.google.com/drive/folders/17Zuue0M3SX36m9dK_jl02gdxPVu_De9f?usp=sharing).

Получившийся датасет доступен на [гугл диске](https://drive.google.com/drive/folders/1Vac7WEmrV-NGRH9je6vXiDHDtw-Upp8f?usp=sharing).

# Создание фотореалистической версии из эскиза:  
Используется генеративно-состязательная сеть, описанная в статье [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf), также называемая pix2pix. Код в репозитории - попытка наиболее точной имплементации архитеркуры из статьи. Используются также значения гиперпараметров, предложенные в статье.

# Комментарии к запуску: 
Возможно два варианта запуска: 1. для обучения, 2. для использования обученной модели. 

1) Для запуска тренировки модели необходим csv файл, где в столбце 'cats' прописаны пути до изображений котов из датасета, а в столбце 'masks' соответствующие им эскизы, в третьем столбце 'kfold' нужно прописать значения, указывающие на принадлежность данных к обучающим или валидационным (например: 'kfold'=5 => данные валидационные, 'kfold'=0 => данные обучающие). Архив с данными и пример такого csv-файла находятся на том же [гугл диске](https://drive.google.com/drive/folders/1Vac7WEmrV-NGRH9je6vXiDHDtw-Upp8f?usp=sharing). Скачать предобученные генератор и дискриминатор можно [здесь](https://drive.google.com/drive/folders/1dh21no-tVoBcDiPDwtoqkCz6KvUnECj7?usp=sharing). 

Запустите trainer.py и на место кавычек введите аргументы в виде: 
```
python trainer.py --DATASET_FILE_PATH '' --LOAD_MODEL '' --CHECKPOINT_DISC '' --CHECKPOINT_GEN '' --EXAMPLES_PATH '' --FOLD ''
```
DATASET_FILE_PATH (str): Путь до csv файла с путями до тренировочных данных

LOAD_MODEL (bool): Загрузить предобученную модель? True, если да; иначе False

CHECKPOINT_DISC (str): Путь для сохранения дискриминатора

CHECKPOINT_GEN (str): Путь для сохранения генератора

EXAMPLES_PATH (str): Путь для сохранения примеров работы генератора

FOLD (int): Валидационный фолд

По дефолту модели сохраняются в metadata/models.

2) Запустите inference.py и на место кавычек введите аргументы в виде: 
```
python inference.py --MODEL_PATH '' --IMAGE_PATH '' --SAVE_PATH '' 
```
MODEL_PATH (str): Путь до предобученного генератора ([скачать генератор](https://drive.google.com/drive/folders/1dh21no-tVoBcDiPDwtoqkCz6KvUnECj7?usp=sharing)). 

IMAGE_PATH (str): Путь до эскиза

SAVE_PATH (str): Путь для сохранения синтезированной фотореалистической версии


#  Кривые обучения (на тренировочных данных):
(График функции потерь не совсем точный, тк значение функции потерь не усреднялось по всем итерациям эпохи, а принималось равным значению функции потерь только на последней итерации эпохи. На данный момент заново обучаю модель, чтобы переделать графики).


![Image alt](https://github.com/adobshik/sketch-to-photo/blob/main/example/gen_testplot.png)
![Image alt](https://github.com/adobshik/sketch-to-photo/blob/main/example/disc_testplot.png)
