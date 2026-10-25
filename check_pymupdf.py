# Диагностика: пробуем вытащить текст через pymupdf вместо pypdf
# Запуск:
#   pip install pymupdf
#   python check_pymupdf.py bank_docs\ALFA-Macro_Insights-Policy_Rate_11Sep.pdf
#
# Если для этого файла пишет "символов текста" > 0 на страницах -> дело в pypdf,
# переключаем bank_rag.py на pymupdf.
# Если тоже везде "(пусто)" -> PDF реально без текстового слоя, нужен OCR.

import fitz  # pymupdf
import sys

path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    print("Использование: python check_pymupdf.py путь\\к\\файлу.pdf")
    sys.exit(1)

doc = fitz.open(path)
print(f"Файл: {path}")
print(f"Страниц: {len(doc)}")

total_chars = 0
for i, page in enumerate(doc, start=1):
    text = page.get_text().strip()
    total_chars += len(text)
    if text:
        print(f"  стр.{i}: {len(text)} символов текста -> {text[:100]!r}")
    else:
        print(f"  стр.{i}: (пусто)")

print()
if total_chars == 0:
    print("ИТОГ: PyMuPDF тоже не нашёл текста -> PDF без текстового слоя (скан/картинки). Нужен OCR.")
else:
    print(f"ИТОГ: PyMuPDF нашёл {total_chars} символов текста -> pypdf не справлялся, переключаемся на pymupdf.")