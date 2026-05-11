from __future__ import annotations

import glob
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import ast

import pandas as pd


ANSWER_COL_CANDIDATES = [
    'top_point_ids'
]

QID_COL_CANDIDATES = ['qid']

CONFIG_COL_CANDIDATES = [
    'indexing',
    'retrieval_strategy',
    'serialization',
    'chunking',
    'query_expansion',
    'encoding_budget',
    'top_k',
    'rerank_k',
    'collection',
    'corpus_vector_names',
    'query_vector_names',
]


@dataclass(frozen=True)
class QuestionRecord:
    article_id: str
    table_id: str
    question_index: int
    qid: str
    question: str
    question_meta: dict[str, Any]
    table_meta: dict[str, Any]
    rag_answers: list[Any]


ConfigSelector = dict[str, Any]


DETAIL_PATHS: list[str | Path] = [
    'review_outputs/qdrant_retrieval_detail_20260507_180308.csv',
]

CONFIG_SELECTOR: ConfigSelector = {
    'indexing': 'hybrid',
    'retrieval_strategy': 'rrf',
    'serialization': 'markdown',
    'chunking': 'table_level',
    'query_expansion': 'none',
    'encoding_budget': 512,
    'top_k': 10,
    'rerank_k': 50,
    'collection': 'Qwen_Qwen3-Embedding-4B_Qdrant_splade_custom',
    'corpus_vector_names': 'dense_markdown_table_level_512|sparse_markdown_table_level_512',
    'query_vector_names': 'dense_query_none|sparse_query_none',
}

QUESTIONS_DIR: str | Path = 'rag_questions'
DATA_DIR: str | Path = 'data'
OUTPUT_PATH: str | Path = 'review_outputs/stage2_eval_data_wiki.jsonl'
OUTPUT_FORMAT: str = 'jsonl'
DETAIL_READ_CHUNK_SIZE: int = 100_000


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _iter_question_files(questions_dir: Path) -> Iterable[Path]:
    return sorted(p for p in questions_dir.rglob('table_*.json') if p.is_file())


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    return value


def _normalize_answer_list(value: Any) -> list[Any]:
    value = _normalize_scalar(value)
    if value is None:
        return []

    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            normalized = _normalize_scalar(item)
            if normalized is not None:
                out.append(normalized)
        return out

    if isinstance(value, tuple):
        return _normalize_answer_list(list(value))

    if isinstance(value, dict):
        for key in ('answer', 'answers', 'text', 'content', 'value', 'prediction', 'pred'):
            if key in value:
                nested = _normalize_answer_list(value[key])
                if nested:
                    return nested
        return [value]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            normalized = _normalize_answer_list(parsed)
            if normalized:
                return normalized

        if '|' in text:
            parts = [part.strip() for part in text.split('|')]
            parts = [part for part in parts if part]
            if len(parts) > 1:
                return parts

        return [text]

    return [value]


def _pick_first_present(row: pd.Series, candidates: list[str]) -> Any:
    for col in candidates:
        if col in row.index:
            value = _normalize_scalar(row[col])
            if value is not None:
                return value
    return None


def _parse_qid_reference(qid: str) -> tuple[str, str, int]:
    qid = str(qid).strip()
    table_ref, separator, question_index_text = qid.partition('::q::')
    if not separator:
        raise ValueError(f'Invalid qid format, expected <article_id>/<table_name>::q::<index>: {qid!r}')

    article_id, slash, table_name = table_ref.partition('/')
    if not slash or not article_id or not table_name:
        raise ValueError(f'Invalid qid table reference, expected <article_id>/<table_name>: {qid!r}')

    try:
        question_index = int(question_index_text)
    except ValueError as e:
        raise ValueError(f'Invalid qid question index: {qid!r}') from e

    return article_id, table_name, question_index


def _config_signature_from_row(row: pd.Series) -> tuple[Any, ...]:
    signature: list[Any] = []
    for col in CONFIG_COL_CANDIDATES:
        if col in row.index:
            signature.append(_normalize_scalar(row[col]))
        else:
            signature.append(None)
    return tuple(signature)


def _normalize_config_value(value: Any) -> Any:
    value = _normalize_scalar(value)
    if isinstance(value, str):
        if '|' in value:
            parts = [part.strip() for part in value.split('|') if part.strip()]
            if len(parts) > 1:
                return parts
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except Exception:
                continue
            return _normalize_config_value(parsed)
    if isinstance(value, list):
        return [_normalize_scalar(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_scalar(item) for item in value]
    return value


def _value_matches_selector(row_value: Any, selector_value: Any) -> bool:
    row_value = _normalize_config_value(row_value)
    selector_value = _normalize_config_value(selector_value)

    if selector_value is None:
        return True

    if isinstance(selector_value, (list, tuple, set)):
        selector_items = [str(item) for item in selector_value]
        if isinstance(row_value, (list, tuple, set)):
            return [str(item) for item in row_value] == selector_items
        return str(row_value) == '|'.join(selector_items)

    if isinstance(row_value, (list, tuple, set)):
        return str(selector_value) == '|'.join(str(item) for item in row_value)

    return str(row_value) == str(selector_value)


def _select_single_config_frame(df: pd.DataFrame, config_selector: ConfigSelector | None = None) -> pd.DataFrame:
    if df.empty:
        return df

    available_cols = [col for col in CONFIG_COL_CANDIDATES if col in df.columns]
    if not available_cols:
        return df

    if not config_selector:
        signatures = df.apply(_config_signature_from_row, axis=1)
        unique_signatures = list(dict.fromkeys(signatures.tolist()))
        if not unique_signatures:
            return df
        if len(unique_signatures) > 1:
            raise ValueError(
                'detail report contains multiple configurations; provide config_selector to pick one explicitly'
            )
        selected_signature = unique_signatures[0]
        mask = signatures == selected_signature
        return df.loc[mask].copy()

    mask = pd.Series(True, index=df.index)
    for col, selector_value in config_selector.items():
        if col not in df.columns:
            continue
        mask &= df[col].apply(lambda row_value: _value_matches_selector(row_value, selector_value))

    return df.loc[mask].copy()


def _detail_usecols(config_selector: ConfigSelector | None = None) -> list[str]:
    usecols = list(dict.fromkeys([*QID_COL_CANDIDATES, *ANSWER_COL_CANDIDATES]))

    if config_selector:
        for col in config_selector:
            if col in CONFIG_COL_CANDIDATES:
                usecols.append(col)
    else:
        usecols.extend(CONFIG_COL_CANDIDATES)

    return list(dict.fromkeys(usecols))


def _iter_filtered_detail_rows(
    detail_path: Path,
    *,
    config_selector: ConfigSelector | None = None,
) -> Iterable[pd.DataFrame]:
    usecols = _detail_usecols(config_selector)

    reader = pd.read_csv(
        detail_path,
        usecols=usecols,
        chunksize=DETAIL_READ_CHUNK_SIZE,
        low_memory=True
    )

    if isinstance(reader, pd.DataFrame):
        chunks = [reader]
    else:
        chunks = reader

    for chunk in chunks:
        if chunk.empty:
            continue
        yield _select_single_config_frame(chunk, config_selector=config_selector)


def _read_detail_report(
    detail_paths: list[Path],
    config_selector: ConfigSelector | None = None,
) -> dict[str, list[Any]]:
    answers_by_qid: dict[str, list[Any]] = OrderedDict()

    for detail_path in detail_paths:
        matched_any = False
        for chunk in _iter_filtered_detail_rows(detail_path, config_selector=config_selector):
            if chunk.empty:
                continue

            qid_col = next((col for col in QID_COL_CANDIDATES if col in chunk.columns), None)
            if qid_col is None:
                raise ValueError(
                    f'No qid column found in {detail_path}. Expected one of: {", ".join(QID_COL_CANDIDATES)}'
                )

            answer_cols = [col for col in ANSWER_COL_CANDIDATES if col in chunk.columns]
            if not answer_cols:
                raise ValueError(
                    f'No answer-like columns found in {detail_path}. Expected one of: {", ".join(ANSWER_COL_CANDIDATES)}'
                )

            matched_any = True
            for _, row in chunk.iterrows():
                qid = _normalize_scalar(row[qid_col])
                if qid is None:
                    continue

                answers = _normalize_answer_list(_pick_first_present(row, answer_cols))
                if not answers:
                    continue

                answers_by_qid[str(qid)] = answers

        if not matched_any:
            expected = ', '.join(f'{k}={v!r}' for k, v in config_selector.items()) if config_selector else '<any>'
            raise ValueError(f'No rows matched config_selector in detail report {detail_path}: {expected}')

    return answers_by_qid


def _build_question_records(
    questions_dir: Path,
    data_dir: Path,
    answers_by_qid: dict[str, list[Any]],
) -> list[QuestionRecord]:
    records: list[QuestionRecord] = []
    questions_cache: dict[Path, list[Any]] = {}
    table_meta_cache: dict[Path, dict[str, Any]] = {}

    for qid, rag_answers in answers_by_qid.items():
        article_id, table_name, question_index = _parse_qid_reference(qid)

        question_file = questions_dir / article_id / f'{table_name}.json'
        table_meta_path = data_dir / article_id / f'{table_name}_meta.json'

        if not question_file.exists():
            raise FileNotFoundError(f'Missing question file: {question_file}')
        if not table_meta_path.exists():
            raise FileNotFoundError(f'Missing table meta file: {table_meta_path}')

        if question_file not in questions_cache:
            questions = _load_json(question_file)
            if not isinstance(questions, list):
                raise ValueError(f'Question file must contain a list: {question_file}')
            questions_cache[question_file] = questions

        if table_meta_path not in table_meta_cache:
            table_meta = _load_json(table_meta_path)
            if not isinstance(table_meta, dict):
                raise ValueError(f'Table meta file must contain a dict: {table_meta_path}')
            table_meta_cache[table_meta_path] = table_meta

        questions = questions_cache[question_file]
        if question_index < 0 or question_index >= len(questions):
            raise IndexError(f'Question index out of range for {qid!r}: {question_index}')

        question_meta = questions[question_index]
        if not isinstance(question_meta, dict):
            raise ValueError(f'Question item must be a dict: {question_file} [index={question_index}]')

        question = str(question_meta.get('question', '')).strip()
        enriched_question_meta = dict(question_meta)
        enriched_question_meta.setdefault('qid', qid)
        enriched_question_meta.setdefault('question_index', question_index)

        records.append(
            QuestionRecord(
                article_id=article_id,
                table_id=f'{article_id}/{table_name}',
                question_index=question_index,
                qid=qid,
                question=question,
                question_meta=enriched_question_meta,
                table_meta=table_meta_cache[table_meta_path],
                rag_answers=rag_answers,
            )
        )

    return records


def _record_to_dict(record: QuestionRecord) -> dict[str, Any]:
    return {
        'article_id': record.article_id,
        'table_id': record.table_id,
        'question_index': record.question_index,
        'qid': record.qid,
        'question': record.question,
        'rag_answers': record.rag_answers,
        'table_meta': record.table_meta,
        'question_meta': record.question_meta,
    }


def _write_output(records: list[QuestionRecord], output_path: Path, output_format: str) -> None:
    payload = [_record_to_dict(record) for record in records]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == 'jsonl':
        with output_path.open('w', encoding='utf-8') as f:
            for item in payload:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return

    if output_format == 'json':
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return

    raise ValueError("output_format must be 'jsonl' or 'json'")


def _parse_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        for raw_item in value.split(','):
            item = raw_item.strip()
            if not item:
                continue
            matches = [Path(p) for p in glob.glob(item)] if any(ch in item for ch in '*?[') else [Path(item)]
            paths.extend(matches)
    return sorted({path.resolve() for path in paths})


def prepare_stage2_eval_data(
    detail_paths: list[str | Path],
    *,
    config_selector: ConfigSelector,
    questions_dir: str | Path = 'rag_questions',
    data_dir: str | Path = 'data',
    output_path: str | Path | None = None,
    output_format: str = 'jsonl',
) -> list[dict[str, Any]]:
    resolved_detail_paths = [Path(path) for path in detail_paths]
    if not resolved_detail_paths:
        raise ValueError('detail_paths must not be empty')

    answers_by_qid = _read_detail_report(resolved_detail_paths, config_selector=config_selector)
    records = _build_question_records(Path(questions_dir), Path(data_dir), answers_by_qid)
    payload = [_record_to_dict(record) for record in records]

    if output_path is not None:
        _write_output(records, Path(output_path), output_format)

    return payload


def main() -> None:
    detail_paths = _parse_paths([str(path) for path in DETAIL_PATHS])
    if not detail_paths:
        raise SystemExit('DETAIL_PATHS is empty')

    answers_by_qid = _read_detail_report(detail_paths, config_selector=CONFIG_SELECTOR)
    records = _build_question_records(Path(QUESTIONS_DIR), Path(DATA_DIR), answers_by_qid)
    _write_output(records, Path(OUTPUT_PATH), OUTPUT_FORMAT)

    missing_answers = sum(1 for record in records if not record.rag_answers)
    print(f'Wrote {len(records)} records to {OUTPUT_PATH}')
    print(f'Records without RAG answers: {missing_answers}')


if __name__ == '__main__':
    main()