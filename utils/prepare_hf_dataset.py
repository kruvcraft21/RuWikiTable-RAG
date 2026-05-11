import os
import json
import pandas as pd
from rich.traceback import install
install(show_locals=True)

def build_and_save_parquet(data_dir, questions_dir, output_file="train.jsonl"):
    rows = []
    for wiki_id in os.listdir(questions_dir):
        q_wiki_path = os.path.join(questions_dir, wiki_id)
        if not os.path.isdir(q_wiki_path): continue
        
        # Мета страницы
        with open(os.path.join(data_dir, wiki_id, "page_meta.json"), 'r') as f:
            page_meta = json.dumps(json.load(f), ensure_ascii=False)

        for q_file in os.listdir(q_wiki_path):
            if 'error' in q_file:
                continue
            prefix = q_file.replace(".json", "")
            
            # Таблица и её мета
            df_table = pd.read_csv(os.path.join(data_dir, wiki_id, f"{prefix}.csv"), sep='|', index_col=0).reset_index(drop=True)
            with open(os.path.join(data_dir, wiki_id, f"{prefix}_meta.json"), 'r') as f:
                table_meta = json.dumps(json.load(f), ensure_ascii=False)
            
            # Вопросы
            with open(os.path.join(q_wiki_path, q_file), 'r') as f:
                questions = json.load(f)
            
            for q in questions:
                rows.append({
                    "question": q["question"],
                    "answer": q["answer"],
                    "table": df_table.to_csv(index=False), # или .to_markdown()
                    "question_type": q.get("question_type", ""),
                    "reasoning": q.get("reasoning", ""),
                    "supporting_rows": [str(row) for row in q.get("supporting_rows", [])],
                    "page_meta": page_meta,
                    "table_meta": table_meta
                })
    
        # Сохраняем в JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in rows:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"Готово! Собрано строк: {len(rows)}")
    print(f"Файл сохранен как: {output_file}")

build_and_save_parquet("data", "rag_questions")
