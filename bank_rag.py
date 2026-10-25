"""
bank_rag.py
Поиск по аналитическим материалам Альфа-Банка (Приложение 2: 9 PDF --
инфляция, ставка, сталь, золото, потребление, макростатистика, FI strategy)
на ЛОКАЛЬНЫХ эмбеддингах (подтверждено организаторами: эмбеддинг-модели
не входят в ограничение по списку генеративных LLM и не тратят бюджет 500р/день).

ВАЖНО: все 9 PDF подтверждённо не содержат текстового слоя (проверено и
pypdf, и PyMuPDF -- оба дают 0 символов на всех страницах). Это растровые
страницы, поэтому текст достаётся через OCR (Tesseract), а не через прямое
извлечение. OCR -- локальный, бесплатный инструмент, в бюджет 500р/день
НЕ входит (это не генеративная LLM).

Установка (один раз):
    pip install sentence-transformers pymupdf pytesseract numpy
    + поставить сам Tesseract OCR (бинарник, не pip-пакет):
      https://github.com/UB-Mannheim/tesseract/wiki
      при установке отметь галочку "Russian" в Additional language data.

Если tesseract.exe НЕ в PATH, пропиши путь к нему в переменной
TESSERACT_CMD ниже (либо через переменную окружения TESSERACT_CMD).

Запуск:
    python bank_rag.py --pdf-dir ./bank_docs --query "ключевая ставка строительство"
"""

import os
import re
import json
import argparse
import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-large"  # можно заменить на -base для скорости
CHUNK_SIZE = 800     # символов на чанк -- грубо, но достаточно для демо
CHUNK_OVERLAP = 150

# OCR-настройки
OCR_DPI = 200          # выше -> точнее распознавание, но медленнее (200 обычно достаточно)
OCR_LANG = "rus+eng"   # какие языки распознавать (должны быть поставлены в Tesseract)

# Путь к tesseract.exe, если он не добавлен в PATH при установке.
# Можно также задать через переменную окружения TESSERACT_CMD.
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)


def _configure_tesseract():
    import pytesseract
    if os.path.exists(TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    return pytesseract


def extract_text_chunks(pdf_path: str, debug: bool = False):
    """
    Извлекает текст постранично через OCR:
    1. Рендерим страницу PDF в картинку через PyMuPDF (fitz).
    2. Прогоняем картинку через Tesseract OCR (pytesseract).
    3. Чанкуем текст страницы, как и раньше, с сохранением номера страницы.
    """
    import fitz  # pymupdf
    pytesseract = _configure_tesseract()

    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
    except Exception as e:
        print(f"  [{os.path.basename(pdf_path)}] НЕ УДАЛОСЬ ОТКРЫТЬ: {e}")
        return []

    if debug:
        print(f"  [{os.path.basename(pdf_path)}] страниц: {num_pages} (режим: OCR)")

    chunks = []
    empty_pages = 0
    zoom = OCR_DPI / 72  # fitz рендерит в 72 dpi по умолчанию, масштабируем
    matrix = fitz.Matrix(zoom, zoom)

    for page_num, page in enumerate(doc, start=1):
        try:
            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("png")
        except Exception as e:
            if debug:
                print(f"    стр.{page_num}: ошибка рендера в картинку: {e}")
            continue

        try:
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(img_bytes))
            raw_text = pytesseract.image_to_string(image, lang=OCR_LANG)
        except Exception as e:
            if debug:
                print(f"    стр.{page_num}: ошибка OCR: {e}")
            continue

        page_text = re.sub(r"\s+", " ", raw_text).strip()

        if not page_text:
            empty_pages += 1
            continue

        start = 0
        while start < len(page_text):
            end = start + CHUNK_SIZE
            chunk_text = page_text[start:end]
            if len(chunk_text) > 50:
                chunks.append({"text": chunk_text, "page": page_num})
            start = end - CHUNK_OVERLAP

        if debug:
            print(f"    стр.{page_num}: {len(page_text)} символов после OCR")

    if debug:
        print(f"  [{os.path.basename(pdf_path)}] пустых страниц: {empty_pages}/{num_pages}, "
              f"чанков получено: {len(chunks)}")
        if chunks:
            print(f"    пример текста (стр.{chunks[0]['page']}): {chunks[0]['text'][:150]!r}")

    return chunks


def build_index(pdf_dir: str, index_path: str = "bank_rag_index.json"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    records = []
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        for i, chunk in enumerate(extract_text_chunks(path, debug=True)):
            records.append({
                "doc": fname,
                "chunk_id": i,
                "page": chunk["page"],
                "text": chunk["text"],
            })

    if not records:
        print("ВНИМАНИЕ: 0 чанков даже после OCR -- индекс не будет построен. "
              "Проверь, что Tesseract установлен и путь TESSERACT_CMD верный "
              "(или tesseract.exe в PATH), и что язык 'rus' реально доступен "
              "(tesseract --list-langs).")
        out = {"model": MODEL_NAME, "records": [], "embeddings": []}
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        return

    # e5 требует префикс "passage: " для документов при индексации
    texts = ["passage: " + r["text"] for r in records]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    out = {
        "model": MODEL_NAME,
        "records": records,
        "embeddings": embeddings.tolist(),
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Проиндексировано чанков: {len(records)} -> {index_path}")


def search(query: str, index_path: str = "bank_rag_index.json", top_k: int = 3):
    from sentence_transformers import SentenceTransformer
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("embeddings"):
        print("Индекс пуст -- сначала построй его через --pdf-dir.")
        return []

    model = SentenceTransformer(data["model"])
    # e5 требует префикс "query: " для поисковых запросов
    q_emb = model.encode(["query: " + query], normalize_embeddings=True)[0]

    emb_matrix = np.array(data["embeddings"])
    scores = emb_matrix @ q_emb
    top_idx = np.argsort(-scores)[:top_k]

    results = []
    for i in top_idx:
        r = data["records"][i]
        results.append({"doc": r["doc"], "chunk_id": r["chunk_id"], "page": r["page"],
                        "score": float(scores[i]), "text": r["text"]})
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", help="папка с PDF Приложения 2, чтобы построить индекс")
    ap.add_argument("--query", help="поисковый запрос, например 'ключевая ставка строительство'")
    ap.add_argument("--index-path", default="bank_rag_index.json")
    args = ap.parse_args()

    if args.pdf_dir:
        build_index(args.pdf_dir, args.index_path)

    if args.query:
        for r in search(args.query, args.index_path):
            print(f"[{r['score']:.3f}] {r['doc']} #{r['chunk_id']}")
            print(r["text"][:300], "...\n")