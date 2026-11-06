# Data

Датасет: [ODIR-5K](https://odir2019.grand-challenge.org/dataset/) (зеркало на Kaggle: `andrewmvd/ocular-disease-recognition-odir5k`).

Данные не хранятся в git. Чтобы скачать датасет в `data/raw/`, получите API-токен Kaggle (kaggle.com → Settings → API) и из корня репозитория запустите:

```bash
bash scripts/download_data.sh
```

| Папка | Содержимое | В git |
|-------|------------|:-----:|
| `raw/` | исходный датасет, не изменяется | нет |
| `processed/` | обработанные изображения, кэш эмбеддингов | нет |
| `splits/` | фиксированные train/val/test CSV (сплит по пациентам) | да |
