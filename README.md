# Bengali PDF -> QA Dataset (CSV) using Groq

এই প্রজেক্ট বাংলা PDF বই থেকে উচ্চমানের প্রশ্ন-উত্তর dataset তৈরি করে।
আউটপুট CSV কলাম:
- `question`
- `answer`
- `chapter`

## কীভাবে কাজ করে (quality-first pipeline)

1. PDF থেকে অধ্যায়ভিত্তিক টেক্সট extraction
- প্রথমে PDF TOC (table of contents) ব্যবহার করে chapter split করার চেষ্টা করে
- TOC না থাকলে heuristic heading detection (`অধ্যায়`, `পরিচ্ছেদ`, `Chapter`) ব্যবহার করে
- extraction শুরুর আগে first-page probe দিয়ে OCR প্রয়োজন কিনা detect করে
- OCR প্রয়োজন না হলে native PDF parsing ব্যবহার করে
- OCR প্রয়োজন হলে OCR extraction ব্যবহার করে

2. Chapter summarization (Groq)
- প্রতিটি অধ্যায়ের সংক্ষিপ্ত সারাংশ তৈরি হয়

3. Vector retrieval grounding
- অধ্যায়ের টেক্সট chunk করা হয়
- multilingual embedding দিয়ে local vector index তৈরি হয়
- প্রশ্ন তৈরি ও উত্তর লেখার সময় relevant chunk retrieve করা হয়

4. Chunk-batch QnA generation (Groq)
- বই/অধ্যায়কে ছোট ছোট chunk এ ভাগ করা হয়
- কয়েকটি chunk একসাথে (chunk group) একটি API call-এ পাঠানো হয়
- একই call থেকে একাধিক QnA pair তৈরি হয়
- এতে API call কমে এবং টোকেন ব্যবহারে নিয়ন্ত্রণ রাখা যায়

5. Quality filtering (Groq self-judge)
- relevance, grounding, clarity স্কোর করে
- threshold pass না করলে বাদ দেওয়া হয়

Rate-limit resilience:

- Groq 429 হলে script auto-wait করে retry করে
- checkpoint file-এ progress save হয়
- `--resume` দিলে আগের অবস্থান থেকে চালিয়ে নেয়
- `--low-token-mode` দিলে প্রতি call-এ context ছোট হয় এবং candidate প্রশ্ন কমে

6. CSV export
- final file: `question,answer,chapter`

## কেন এই Groq model

ডিফল্ট: `meta-llama/llama-4-scout-17b-16e-instruct`

কারণ:
- দীর্ঘ context handling ভাল
- বাংলা generation quality সাধারণত স্থিতিশীল
- structured JSON output workflow এ ভালো কাজ করে

চাইলে `--model` দিয়ে অন্য Groq model দিতে পারবেন।

## Setup

```powershell
cd c:\sambhranta\projects\WBG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

যদি PDF scanned/image-based হয়, তাহলে Tesseract OCR install করুন (Windows):

1. Tesseract installer install করুন
2. Bengali trained data (`ben`) enabled আছে কিনা নিশ্চিত করুন
3. Tesseract path system PATH-এ যোগ করুন
4. নতুন terminal খুলে verify করুন: `tesseract --version`

Recommended OCR setup (best for Bengali):

- Tesseract 5
- `tessdata_best` থেকে `ben.traineddata`
- OCR language: `ben+eng`

Windows quick install:

```powershell
winget install -e --id tesseract-ocr.tesseract --accept-source-agreements --accept-package-agreements
tesseract --list-langs
```

`--list-langs` এ `ben` না দেখালে `tessdata_best` থেকে `ben.traineddata` এনে `TESSDATA_PREFIX` path-এ রাখুন।

`.env` এ path config করতে পারেন:

```env
TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
TESSDATA_PREFIX=C:\\Program Files\\Tesseract-OCR\\tessdata_best
```

`.env.example` কপি করে `.env` বানান এবং API key দিন:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Run

```powershell
python build_bengali_qa_dataset.py `
  --pdf "path\to\your_bengali_book.pdf" `
  --output "bengali_qa_dataset.csv" `
  --questions-per-chapter 12 `
  --chunk-group-size 3 `
  --qa-pairs-per-call 2 `
  --quality-threshold 0.75
```

OCR settings (optional):

```powershell
python build_bengali_qa_dataset.py --pdf "book.pdf" --ocr-language "ben+eng"
python build_bengali_qa_dataset.py --pdf "book.pdf" --disable-ocr-fallback
python build_bengali_qa_dataset.py --pdf "book.pdf" --ocr-language "ben+eng" --ocr-oem 1 --ocr-psm 3
python build_bengali_qa_dataset.py --pdf "book.pdf" --ocr-probe-pages 10
```

Large-book / rate-limit friendly run:

```powershell
python build_bengali_qa_dataset.py `
  --pdf "book.pdf" `
  --output "bengali_qa_dataset.csv" `
  --start-chapter 1 `
  --max-chapters 20 `
  --questions-per-chapter 1 `
  --low-token-mode `
  --skip-quality-judge `
  --resume
```

পরের ব্যাচ চালাতে শুধু `--start-chapter` বাড়ান (যেমন 21, 41, ...), এবং একই `--checkpoint-path` রাখুন।

## Useful tuning

- আরো বেশি coverage চাইলে:
  - `--questions-per-chapter` বাড়ান
  - `--chunk-group-size` 3-5 রাখুন
  - `--qa-pairs-per-call` 2-3 রাখুন
- hallucination কমাতে:
  - `--quality-threshold` বাড়ান (যেমন `0.78`)
  - `--top-k` 4-6 রাখুন
- context coherence improve করতে:
  - `--chunk-size` 900-1300 range টেস্ট করুন

## Optional production improvement

- local in-memory vector index এর জায়গায় persistent vector DB ব্যবহার করতে পারেন:
  - Chroma
  - Qdrant
  - Milvus

এতে:
- পুনরায় রান দ্রুত হয়
- chapter-level indexing persist করা যায়
- audit/debug সহজ হয়

## Output format example

```csv
question,answer,chapter
"প্রশ্ন ১...","উত্তর ১...","অধ্যায় ১"
"প্রশ্ন ২...","উত্তর ২...","অধ্যায় ১"
```
