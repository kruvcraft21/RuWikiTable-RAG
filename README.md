# RuWikiTable-RAG

Проект для подготовки, индексации и оценки таблиц из русскоязычной Википедии в RAG-пайплайне.
Репозиторий собран вокруг ноутбуков для подготовки данных, загрузки эмбеддингов в Qdrant, retrieval/eval и генерации ответов на втором этапе.

## Что внутри

- `create_dataset.ipynb` — подготовка исходного набора данных.
- `qdrant_push_embd.ipynb` — загрузка эмбеддингов и индексация в Qdrant.
- `qdrant_retrieve_evaluate.ipynb` — retrieval и оценка качества поиска.
- `qdrant_results_analysis.ipynb` — разбор результатов экспериментов.
- `stage2_answer_generation.ipynb` — генерация ответов на втором этапе.
- `stage2_eval.ipynb` — оценка второго этапа.
- `utils/prepare_hf_dataset.py` — сборка датасета в JSONL для обучения или дальнейшей обработки.
- `utils/prepare_stage2_eval_data.py` — подготовка данных для stage 2 eval из detail-отчётов и разметки вопросов.

## Требования

- Python 3.11+.
- Docker и Docker Compose.
- Зависимости из `requirements.txt`.

## Быстрый старт

1. Создайте виртуальное окружение и установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2. Поднимите Qdrant:

```bash
docker compose up -d
```

3. Подготовьте данные и запустите нужный этап пайплайна в ноутбуках или через скрипты из `utils/`.

## Типовой пайплайн

1. Сформировать исходный датасет и структуру `data/` и `rag_questions/`.
2. Собрать обучающий/служебный JSONL через `utils/prepare_hf_dataset.py`.
3. Загрузить векторный индекс в Qdrant через `qdrant_push_embd.ipynb`.
4. Запустить retrieval и посмотреть метрики в `qdrant_retrieve_evaluate.ipynb`.
5. Сгенерировать ответы на stage 2 в `stage2_answer_generation.ipynb`.
6. Подготовить и посчитать оценку stage 2 в `utils/prepare_stage2_eval_data.py` и `stage2_eval.ipynb`.

## Данные и выходные файлы

Проект ожидает локальные каталоги вроде `data/`, `rag_questions/`, `review_outputs/` и `qdrant_storage/`.
Скрипты в `utils/` читают метаданные таблиц, вопросы и detail-отчёты, а затем пишут результат в JSON или JSONL.

## Примечания

- В `docker-compose.yml` уже задан сервис Qdrant и включён API key.
- Некоторые ноутбуки и скрипты рассчитаны на конкретную структуру файлов, поэтому лучше сохранять текущие имена каталогов и артефактов.

