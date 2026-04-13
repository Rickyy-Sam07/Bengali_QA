import argparse
import csv
import io
import json
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fitz
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@dataclass
class Chapter:
    title: str
    text: str
    start_page: int
    end_page: int


def clean_text(text: str) -> str:
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_chunks(text: str, chunk_size: int = 1100, overlap: int = 180) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("।", start, end),
                text.rfind("\n", start, end),
                text.rfind(".", start, end),
            )
            if boundary != -1 and boundary > start + 200:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks


def group_chunks(chunks: List[str], group_size: int, stride: int) -> List[List[str]]:
    if not chunks:
        return []

    gsize = max(1, group_size)
    step = max(1, stride)
    groups: List[List[str]] = []

    i = 0
    while i < len(chunks):
        group = chunks[i : i + gsize]
        if not group:
            break
        groups.append(group)
        if i + gsize >= len(chunks):
            break
        i += step

    return groups


class PDFChapterExtractor:
    def __init__(
        self,
        pdf_path: str,
        enable_ocr_fallback: bool = True,
        ocr_language: str = "ben+eng",
        ocr_dpi: int = 250,
        ocr_oem: int = 1,
        ocr_psm: int = 3,
        ocr_probe_pages: int = 8,
    ):
        self.pdf_path = pdf_path
        self.enable_ocr_fallback = enable_ocr_fallback
        self.ocr_language = ocr_language
        self.ocr_dpi = ocr_dpi
        self.ocr_oem = ocr_oem
        self.ocr_psm = ocr_psm
        self.ocr_probe_pages = max(1, ocr_probe_pages)

    def _read_pages(self, max_pages: Optional[int] = None) -> List[str]:
        doc = fitz.open(self.pdf_path)
        pages = []
        limit = len(doc) if max_pages is None else min(len(doc), max_pages)
        for i in range(limit):
            pages.append(clean_text(doc[i].get_text("text")))
        doc.close()
        return pages

    def _should_use_ocr(self, pages: List[str]) -> bool:
        if not pages:
            return True

        total_chars = sum(len(p) for p in pages)
        non_empty_pages = sum(1 for p in pages if len(p.strip()) >= 30)
        non_empty_ratio = non_empty_pages / max(1, len(pages))

        bengali_chars = sum(len(re.findall(r"[\u0980-\u09FF]", p)) for p in pages)
        bengali_ratio = bengali_chars / max(1, total_chars)

        too_little_text = total_chars < 1200
        mostly_empty = non_empty_ratio < 0.2
        likely_bad_native = total_chars < 4000 and bengali_ratio < 0.04

        return too_little_text or mostly_empty or likely_bad_native

    def _decide_extraction_mode(self) -> str:
        if not self.enable_ocr_fallback:
            print("ℹ️ OCR fallback disabled. Native PDF parsing ব্যবহার করা হবে।")
            return "native"

        probe_pages = self._read_pages(max_pages=self.ocr_probe_pages)
        if self._should_use_ocr(probe_pages):
            print("ℹ️ OCR প্রয়োজন detected হয়েছে। OCR extraction ব্যবহার করা হবে।")
            return "ocr"

        print("ℹ️ OCR প্রয়োজন নেই। Native PDF parsing ব্যবহার করা হবে।")
        return "native"

    def _read_pages_with_ocr(self) -> List[str]:
        try:
            import pytesseract
            from PIL import Image
        except Exception as exc:
            raise RuntimeError(
                "OCR fallback চালাতে pytesseract ও pillow লাগবে। requirements install করুন।"
            ) from exc

        env_tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
        if env_tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = env_tesseract_cmd

        configured_cmd = pytesseract.pytesseract.tesseract_cmd
        resolved_tesseract = None
        if configured_cmd:
            if os.path.isabs(configured_cmd) and os.path.exists(configured_cmd):
                resolved_tesseract = configured_cmd
            else:
                resolved_tesseract = shutil.which(configured_cmd)

        if resolved_tesseract is None:
            raise RuntimeError(
                "Tesseract OCR executable পাওয়া যায়নি। Windows এ Tesseract install করে PATH এ যোগ করুন।"
            )

        try:
            langs = {lang.strip() for lang in pytesseract.get_languages(config="") if lang.strip()}
        except Exception:
            langs = set()

        requested_langs = [lang.strip() for lang in self.ocr_language.split("+") if lang.strip()]
        if "ben" in requested_langs and langs and "ben" not in langs:
            raise RuntimeError(
                "Bengali OCR model (ben.traineddata) পাওয়া যায়নি। "
                "tessdata_best থেকে ben model install করুন এবং TESSDATA_PREFIX সেট করুন।"
            )

        tessdata_prefix = os.getenv("TESSDATA_PREFIX", "").strip()
        ocr_config = f"--oem {self.ocr_oem} --psm {self.ocr_psm}"
        if tessdata_prefix:
            ocr_config += f" --tessdata-dir {tessdata_prefix}"

        print(f"ℹ️ OCR config: lang={self.ocr_language}, oem={self.ocr_oem}, psm={self.ocr_psm}")
        if tessdata_prefix:
            print(f"ℹ️ Using TESSDATA_PREFIX: {tessdata_prefix}")

        # Quick sanity-check command output before processing all pages.
        try:
            subprocess.run([resolved_tesseract, "--version"], check=True, capture_output=True, text=True)
        except Exception:
            raise RuntimeError("Tesseract run করা যাচ্ছে না। installation/path configuration চেক করুন।")

        doc = fitz.open(self.pdf_path)
        pages: List[str] = []
        zoom = max(1.0, float(self.ocr_dpi) / 72.0)

        for i in tqdm(range(len(doc)), desc="OCR pages", leave=False):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img, lang=self.ocr_language, config=ocr_config)
            pages.append(clean_text(ocr_text))

        doc.close()
        return pages

    def _toc_chapters(self, pages: List[str]) -> List[Chapter]:
        doc = fitz.open(self.pdf_path)
        toc = doc.get_toc(simple=True)
        doc.close()

        if not toc:
            return []

        # Prefer top-level headings. If unavailable, fall back to all TOC entries.
        top = [item for item in toc if len(item) >= 3 and item[0] == 1]
        entries = top if top else [item for item in toc if len(item) >= 3]
        if not entries:
            return []

        chapters: List[Chapter] = []
        for idx, entry in enumerate(entries):
            _, title, start_page = entry
            s = max(1, int(start_page))
            e = len(pages)
            if idx + 1 < len(entries):
                e = max(s, int(entries[idx + 1][2]) - 1)

            text = clean_text("\n\n".join(pages[s - 1 : e]))
            if len(text) < 250:
                continue

            chapters.append(
                Chapter(
                    title=clean_text(str(title)) or f"অধ্যায় {idx + 1}",
                    text=text,
                    start_page=s,
                    end_page=e,
                )
            )

        return chapters

    def _heuristic_chapters(self, pages: List[str]) -> List[Chapter]:
        heading_re = re.compile(
            r"^(?:\s*)(অধ্যায়|পরিচ্ছেদ|Chapter|CHAPTER)\s*([০-৯0-9IVXivx\-:\. ]*)",
            flags=re.IGNORECASE,
        )

        detected: List[Tuple[int, str]] = []
        for page_idx, page_text in enumerate(pages, start=1):
            lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
            for line in lines[:8]:
                if heading_re.search(line):
                    detected.append((page_idx, clean_text(line)))
                    break

        if not detected:
            full = clean_text("\n\n".join(pages))
            return [Chapter(title="সম্পূর্ণ বই", text=full, start_page=1, end_page=len(pages))]

        chapters: List[Chapter] = []
        for i, (start_page, title) in enumerate(detected):
            end_page = len(pages)
            if i + 1 < len(detected):
                end_page = max(start_page, detected[i + 1][0] - 1)

            text = clean_text("\n\n".join(pages[start_page - 1 : end_page]))
            if len(text) < 250:
                continue
            chapters.append(Chapter(title=title, text=text, start_page=start_page, end_page=end_page))

        return chapters

    def extract(self) -> List[Chapter]:
        mode = self._decide_extraction_mode()
        if mode == "ocr":
            pages = self._read_pages_with_ocr()
        else:
            pages = self._read_pages()

        chapters = self._toc_chapters(pages)
        if chapters:
            return chapters
        return self._heuristic_chapters(pages)


class LocalVectorIndex:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def query(self, query_text: str, texts: List[str], embeddings: np.ndarray, top_k: int = 4) -> List[str]:
        q = self.embed([query_text])[0]
        scores = embeddings @ q
        top_ids = np.argsort(-scores)[:top_k]
        return [texts[i] for i in top_ids]


def safe_json_from_text(text: str) -> Dict:
    text = text.strip()

    # Remove Markdown code fences if the model wrapped JSON in them.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the largest JSON-looking object.
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Valid JSON পাওয়া যায়নি।")


class GroqQAGenerator:
    def __init__(self, api_key: str, model: str, rate_limit_retries: int = 5, low_token_mode: bool = False):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.rate_limit_retries = max(0, rate_limit_retries)
        self.low_token_mode = low_token_mode
        self.use_strict_json = not low_token_mode

        # Token-saving profile for quota-constrained runs.
        self.summary_input_chars = 9000 if low_token_mode else 18000
        self.summary_max_tokens = 650 if low_token_mode else 1200
        self.key_points_context_chunks = 4 if low_token_mode else 8
        self.key_points_points_range = "8-12" if low_token_mode else "12-18"
        self.key_points_max_tokens = 850 if low_token_mode else 1400
        self.question_points_limit = 12 if low_token_mode else 24
        self.question_max_tokens = 950 if low_token_mode else 1700
        self.answer_max_tokens = 600 if low_token_mode else 900
        self.judge_max_tokens = 260 if low_token_mode else 400
        self.batch_context_max_chars = 7000 if low_token_mode else 14000
        self.batch_gen_max_tokens = 950 if low_token_mode else 1700

    @staticmethod
    def _extract_retry_wait_seconds(error_text: str) -> Optional[float]:
        text = error_text or ""
        m = re.search(r"try again in\s+(\d+)m([\d\.]+)s", text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))

        m = re.search(r"try again in\s+([\d\.]+)s", text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))

        return None

    def _chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> Dict:
        attempts = self.rate_limit_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                req = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if self.use_strict_json:
                    req["response_format"] = {"type": "json_object"}

                resp = self.client.chat.completions.create(**req)
                return safe_json_from_text(resp.choices[0].message.content or "{}")
            except Exception as exc:
                error_text = str(exc)

                # Some models occasionally fail structured output validation.
                # Fallback to plain completion and parse JSON manually.
                if "json_validate_failed" in error_text or "Failed to generate JSON" in error_text:
                    if self.use_strict_json:
                        self.use_strict_json = False
                        print("⚠️ Structured JSON failed. Relaxed JSON parsing mode enabled.")
                    try:
                        retry_user_prompt = (
                            user_prompt
                            + "\n\nঅবশ্যই কেবল একটি বৈধ JSON object দাও। অতিরিক্ত টেক্সট, markdown, ব্যাখ্যা দেবে না।"
                        )
                        raw_resp = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": retry_user_prompt},
                            ],
                            temperature=max(0.0, min(temperature, 0.2)),
                            max_tokens=max_tokens,
                        )
                        return safe_json_from_text(raw_resp.choices[0].message.content or "{}")
                    except Exception as fallback_exc:
                        print(f"⚠️ JSON fallback parsing failed. Empty payload ব্যবহার করা হবে: {fallback_exc}")
                        return {}

                is_rate_limit = "rate limit" in error_text.lower() or "429" in error_text
                if not is_rate_limit or attempt >= attempts:
                    raise

                wait_s = self._extract_retry_wait_seconds(error_text)
                if wait_s is None:
                    wait_s = min(90.0, 8.0 * attempt)
                wait_s = max(2.0, wait_s + 1.0)
                print(f"⏳ Groq rate limit hit. {wait_s:.1f} সেকেন্ড পরে retry ({attempt}/{attempts})...")
                time.sleep(wait_s)

        raise RuntimeError("Groq request failed unexpectedly.")

    def summarize(self, chapter_title: str, chapter_text: str) -> str:
        payload = self._chat_json(
            system_prompt=(
                "তুমি বাংলা একাডেমিক সম্পাদক। শুধুই বাংলা ভাষায় উত্তর দাও। "
                "আউটপুট অবশ্যই বৈধ JSON হবে।"
            ),
            user_prompt=(
                "নিচের অধ্যায়ের নির্ভুল, সংক্ষিপ্ত কিন্তু তথ্যপূর্ণ সারাংশ দাও। "
                "নতুন তথ্য যোগ করবে না।\n\n"
                f"অধ্যায়: {chapter_title}\n\n"
                f"পাঠ্য:\n{chapter_text[:self.summary_input_chars]}\n\n"
                "JSON format:\n"
                "{\"summary\": \"...\"}"
            ),
            temperature=0.1,
            max_tokens=self.summary_max_tokens,
        )
        return clean_text(payload.get("summary", ""))

    def key_points(self, chapter_title: str, chapter_summary: str, sample_chunks: List[str]) -> List[str]:
        context = "\n\n".join(sample_chunks[: self.key_points_context_chunks])
        payload = self._chat_json(
            system_prompt=(
                "তুমি বাংলা কনটেন্ট কিউরেটর। উত্তর কেবল বাংলা ভাষায় হবে এবং JSON ফরম্যাটে হবে।"
            ),
            user_prompt=(
                f"সারাংশ ও কনটেক্সট দেখে অধ্যায়ের সবচেয়ে গুরুত্বপূর্ণ {self.key_points_points_range}টি তথ্যবিন্দু দাও। "
                "প্রতিটি পয়েন্ট ছোট, স্পষ্ট, এবং বাস্তবভিত্তিক হওয়া চাই।\n\n"
                f"অধ্যায়: {chapter_title}\n"
                f"সারাংশ: {chapter_summary}\n\n"
                f"কনটেক্সট:\n{context}\n\n"
                "JSON format:\n"
                "{\"points\": [\"...\", \"...\"]}"
            ),
            temperature=0.2,
            max_tokens=self.key_points_max_tokens,
        )
        points = payload.get("points", [])
        return [clean_text(p) for p in points if isinstance(p, str) and clean_text(p)]

    def generate_questions(self, chapter_title: str, chapter_summary: str, points: List[str], n: int) -> List[str]:
        points_text = "\n".join([f"- {p}" for p in points[: self.question_points_limit]])
        payload = self._chat_json(
            system_prompt=(
                "তুমি উচ্চমানের বাংলা প্রশ্নপ্রণেতা। প্রশ্নগুলো বাংলা ভাষায়, স্বচ্ছ, প্রাসঙ্গিক, "
                "পুনরাবৃত্তিহীন এবং কনটেক্সট-ভিত্তিক হবে। আউটপুট JSON।"
            ),
            user_prompt=(
                f"অধ্যায়: {chapter_title}\n"
                f"সারাংশ: {chapter_summary}\n\n"
                f"মূল তথ্যবিন্দু:\n{points_text}\n\n"
                f"মোট {n}টি প্রশ্ন তৈরি করো।\n"
                "শর্ত:\n"
                "1) প্রশ্নগুলো কেবল প্রদত্ত তথ্যের উপর ভিত্তি করে হবে\n"
                "2) সহজ-মাঝারি-কঠিন মিশ্রণ থাকবে\n"
                "3) একই প্রশ্নের ভিন্ন রূপ হবে না\n"
                "4) সব প্রশ্ন বাংলা ভাষায় হবে\n\n"
                "JSON format:\n"
                "{\"questions\": [\"...\", \"...\"]}"
            ),
            temperature=0.35,
            max_tokens=self.question_max_tokens,
        )
        questions = payload.get("questions", [])
        return [clean_text(q) for q in questions if isinstance(q, str) and clean_text(q)]

    def answer_question(self, chapter_title: str, question: str, retrieved_context: List[str], chapter_summary: str) -> str:
        context = "\n\n".join(retrieved_context)
        payload = self._chat_json(
            system_prompt=(
                "তুমি বাংলা প্রশ্নোত্তর বিশেষজ্ঞ। কেবল প্রাপ্ত কনটেক্সটের ভিত্তিতে উত্তর দাও। "
                "যদি কনটেক্সটে উত্তর না থাকে, তা স্পষ্টভাবে বলো। আউটপুট JSON।"
            ),
            user_prompt=(
                f"অধ্যায়: {chapter_title}\n"
                f"প্রশ্ন: {question}\n"
                f"অধ্যায়ের সারাংশ: {chapter_summary}\n\n"
                f"রিট্রিভড কনটেক্সট:\n{context}\n\n"
                "নিয়ম:\n"
                "1) উত্তর সম্পূর্ণ বাংলা ভাষায়\n"
                "2) নির্ভুল, সংক্ষিপ্ত, কিন্তু সম্পূর্ণ\n"
                "3) কনটেক্সটের বাইরে তথ্য যোগ করবে না\n\n"
                "JSON format:\n"
                "{\"answer\": \"...\"}"
            ),
            temperature=0.15,
            max_tokens=self.answer_max_tokens,
        )
        return clean_text(payload.get("answer", ""))

    def judge_quality(self, question: str, answer: str, retrieved_context: List[str]) -> Tuple[float, str]:
        context = "\n\n".join(retrieved_context)
        payload = self._chat_json(
            system_prompt=(
                "তুমি বাংলা QA quality reviewer। প্রশ্ন-উত্তর জুটি কনটেক্সট-ভিত্তিক কিনা যাচাই করবে। "
                "আউটপুট JSON।"
            ),
            user_prompt=(
                f"প্রশ্ন: {question}\n"
                f"উত্তর: {answer}\n\n"
                f"কনটেক্সট:\n{context}\n\n"
                "১ থেকে ৫ স্কেলে স্কোর দাও (৫ সর্বোচ্চ)।\n"
                "স্কোরিং ফোকাস: প্রাসঙ্গিকতা, নির্ভুলতা, কনটেক্সট-গ্রাউন্ডিং, ভাষার স্বচ্ছতা।\n\n"
                "JSON format:\n"
                "{\"score\": 4.5, \"reason\": \"...\"}"
            ),
            temperature=0.0,
            max_tokens=self.judge_max_tokens,
        )

        raw_score = payload.get("score", 0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(5.0, score))
        return score / 5.0, clean_text(str(payload.get("reason", "")))

    def generate_qa_pairs_from_chunks(self, chapter_title: str, chunk_group: List[str], n_pairs: int) -> List[Dict[str, str]]:
        if not chunk_group or n_pairs <= 0:
            return []

        context = "\n\n".join([f"[Chunk {i + 1}]\n{c}" for i, c in enumerate(chunk_group)])
        context = context[: self.batch_context_max_chars]

        payload = self._chat_json(
            system_prompt=(
                "তুমি বাংলা প্রশ্নোত্তর বিশেষজ্ঞ। কেবল দেওয়া কনটেক্সটের ভিত্তিতে প্রশ্ন-উত্তর তৈরি করবে। "
                "আউটপুট অবশ্যই JSON object হবে।"
            ),
            user_prompt=(
                f"অধ্যায়: {chapter_title}\n\n"
                "নিচের কনটেক্সট থেকে উচ্চমানের প্রশ্ন-উত্তর তৈরি করো।\n"
                f"মোট {n_pairs}টি QnA pair দাও।\n"
                "নিয়ম:\n"
                "1) ভাষা সম্পূর্ণ বাংলা\n"
                "2) প্রশ্ন স্পষ্ট ও প্রাসঙ্গিক\n"
                "3) উত্তর কনটেক্সট-গ্রাউন্ডেড, অতিরিক্ত কল্পিত তথ্য নয়\n"
                "4) প্রশ্নগুলো একে অপরের পুনরাবৃত্তি হবে না\n\n"
                f"কনটেক্সট:\n{context}\n\n"
                "JSON format:\n"
                "{\"qa_pairs\": [{\"question\": \"...\", \"answer\": \"...\"}]}"
            ),
            temperature=0.2,
            max_tokens=self.batch_gen_max_tokens,
        )

        pairs = payload.get("qa_pairs", [])
        results: List[Dict[str, str]] = []
        if not isinstance(pairs, list):
            return results

        for item in pairs:
            if not isinstance(item, dict):
                continue
            q = clean_text(str(item.get("question", "")))
            a = clean_text(str(item.get("answer", "")))
            if q and a:
                results.append({"question": q, "answer": a})

        return results


def normalize_question(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[\s\t\n]+", " ", t)
    t = re.sub(r"[\?\!\.,;:'\"\-\(\)\[\]{}]", "", t)
    return t


def save_checkpoint(
    checkpoint_path: str,
    rows: List[Dict[str, str]],
    seen_questions: set,
    completed_chapters: set,
) -> None:
    payload = {
        "rows": rows,
        "seen_questions": sorted(seen_questions),
        "completed_chapters": sorted(completed_chapters),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_checkpoint(checkpoint_path: str) -> Tuple[List[Dict[str, str]], set, set]:
    if not os.path.exists(checkpoint_path):
        return [], set(), set()

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("rows", [])
    seen_questions = set(payload.get("seen_questions", []))
    completed_chapters = set(payload.get("completed_chapters", []))
    return rows, seen_questions, completed_chapters


def build_dataset(
    pdf_path: str,
    output_csv: str,
    groq_model: str,
    embedding_model: str,
    questions_per_chapter: int,
    chunk_size: int,
    chunk_overlap: int,
    chunk_group_size: int,
    chunk_group_stride: int,
    qa_pairs_per_call: int,
    top_k: int,
    quality_threshold: float,
    min_chapter_chars: int,
    enable_ocr_fallback: bool,
    ocr_language: str,
    ocr_oem: int,
    ocr_psm: int,
    ocr_probe_pages: int,
    start_chapter: int,
    max_chapters: int,
    skip_quality_judge: bool,
    checkpoint_path: str,
    resume: bool,
    rate_limit_retries: int,
    low_token_mode: bool,
    seed: int,
) -> None:
    random.seed(seed)
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY পাওয়া যায়নি। .env ফাইলে সেট করুন।")

    extractor = PDFChapterExtractor(
        pdf_path=pdf_path,
        enable_ocr_fallback=enable_ocr_fallback,
        ocr_language=ocr_language,
        ocr_oem=ocr_oem,
        ocr_psm=ocr_psm,
        ocr_probe_pages=ocr_probe_pages,
    )
    chapters = extractor.extract()
    if not chapters:
        raise ValueError("PDF থেকে কোনো অধ্যায় পাওয়া যায়নি।")

    usable_chapters = [c for c in chapters if len(clean_text(c.text)) >= min_chapter_chars]
    if not usable_chapters:
        total_chars = sum(len(clean_text(c.text)) for c in chapters)
        raise ValueError(
            "PDF থেকে ব্যবহারযোগ্য টেক্সট পাওয়া যায়নি। "
            f"মোট extracted chars: {total_chars}. "
            "সম্ভবত এটি scanned/image PDF। OCR fallback on আছে কিনা দেখুন, "
            "না হলে Tesseract + Bengali language data install করুন।"
        )

    start_index = max(0, start_chapter - 1)
    end_index = len(usable_chapters)
    if max_chapters > 0:
        end_index = min(end_index, start_index + max_chapters)
    selected_chapters = usable_chapters[start_index:end_index]

    if not selected_chapters:
        raise ValueError("নির্বাচিত chapter range-এ কোনো অধ্যায় নেই। --start-chapter/--max-chapters চেক করুন।")

    vector_index = LocalVectorIndex(embedding_model)
    qa_model = GroqQAGenerator(
        api_key=api_key,
        model=groq_model,
        rate_limit_retries=rate_limit_retries,
        low_token_mode=low_token_mode,
    )

    rows: List[Dict[str, str]] = []
    seen_questions = set()
    completed_chapters = set()
    if resume:
        rows, seen_questions, completed_chapters = load_checkpoint(checkpoint_path)
        print(
            f"ℹ️ Resume enabled: existing rows={len(rows)}, completed chapters={len(completed_chapters)}"
        )

    print(f"মোট অধ্যায় ধরা পড়েছে: {len(chapters)}")
    print(f"ব্যবহারযোগ্য অধ্যায়: {len(usable_chapters)}")
    print(f"প্রসেস হবে chapter #{start_index + 1} থেকে #{end_index} (মোট {len(selected_chapters)})")
    if low_token_mode:
        print("ℹ️ Low-token mode enabled: ছোট context + কম candidate প্রশ্ন ব্যবহার হবে।")

    for chapter_idx, chapter in enumerate(tqdm(selected_chapters, desc="Chapters"), start=start_index + 1):
        chapter_key = f"{chapter_idx}:{chapter.start_page}:{chapter.end_page}:{chapter.title}"
        if chapter_key in completed_chapters:
            continue

        chapter_text = clean_text(chapter.text)
        if len(chapter_text) < min_chapter_chars:
            completed_chapters.add(chapter_key)
            save_checkpoint(checkpoint_path, rows, seen_questions, completed_chapters)
            continue

        chunks = split_chunks(chapter_text, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            completed_chapters.add(chapter_key)
            save_checkpoint(checkpoint_path, rows, seen_questions, completed_chapters)
            continue

        chunk_batches = group_chunks(chunks, group_size=chunk_group_size, stride=chunk_group_stride)
        if not chunk_batches:
            completed_chapters.add(chapter_key)
            save_checkpoint(checkpoint_path, rows, seen_questions, completed_chapters)
            continue

        embeddings = None
        if not skip_quality_judge:
            embeddings = vector_index.embed(chunks)

        try:
            chapter_kept = 0

            for batch in chunk_batches:
                if chapter_kept >= questions_per_chapter:
                    break

                remaining = questions_per_chapter - chapter_kept
                req_pairs = min(max(1, qa_pairs_per_call), max(1, remaining + 1))
                qa_pairs = qa_model.generate_qa_pairs_from_chunks(chapter.title, batch, req_pairs)

                for qa in qa_pairs:
                    if chapter_kept >= questions_per_chapter:
                        break

                    question = clean_text(qa.get("question", ""))
                    answer = clean_text(qa.get("answer", ""))
                    if not question or not answer:
                        continue

                    q_norm = normalize_question(question)
                    if not q_norm or q_norm in seen_questions:
                        continue

                    if not skip_quality_judge and embeddings is not None:
                        query_top_k = min(top_k, len(chunks))
                        if low_token_mode:
                            query_top_k = min(query_top_k, 2)
                        ctx = vector_index.query(question, chunks, embeddings, top_k=query_top_k)
                        score, _ = qa_model.judge_quality(question, answer, ctx)
                        if score < quality_threshold:
                            continue

                    rows.append({"question": question, "answer": answer, "chapter": chapter.title})
                    seen_questions.add(q_norm)
                    chapter_kept += 1
        except Exception as chapter_exc:
            print(f"⚠️ chapter #{chapter_idx} প্রসেসে ত্রুটি, skip করা হলো: {chapter_exc}")

        completed_chapters.add(chapter_key)
        save_checkpoint(checkpoint_path, rows, seen_questions, completed_chapters)

    if not rows:
        raise ValueError("কোনো মানসম্মত প্রশ্ন-উত্তর জুটি তৈরি হয়নি।")

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "chapter"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ CSV তৈরি হয়েছে: {output_csv}")
    print(f"✅ মোট QA জুটি: {len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="বাংলা PDF থেকে উচ্চমানের QA dataset (CSV) তৈরি করুন")
    parser.add_argument("--pdf", required=True, help="ইনপুট বাংলা PDF ফাইলের পাথ")
    parser.add_argument("--output", default="bengali_qa_dataset.csv", help="আউটপুট CSV ফাইলের নাম")
    parser.add_argument("--model", default=DEFAULT_GROQ_MODEL, help="Groq model name")
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer multilingual embedding model",
    )
    parser.add_argument("--questions-per-chapter", type=int, default=10, help="প্রতি অধ্যায়ে টার্গেট প্রশ্ন সংখ্যা")
    parser.add_argument("--chunk-size", type=int, default=1100, help="প্রতি টেক্সট chunk এর ক্যারেক্টার সাইজ")
    parser.add_argument("--chunk-overlap", type=int, default=180, help="chunk overlap")
    parser.add_argument("--chunk-group-size", type=int, default=3, help="একটি API call-এ কতটি chunk যাবে")
    parser.add_argument("--chunk-group-stride", type=int, default=3, help="chunk group slide step")
    parser.add_argument("--qa-pairs-per-call", type=int, default=2, help="প্রতি API call-এ target QnA pair সংখ্যা")
    parser.add_argument("--top-k", type=int, default=4, help="প্রতি প্রশ্নে retrieval context count")
    parser.add_argument("--min-chapter-chars", type=int, default=300, help="অধ্যায় ব্যবহারের জন্য minimum character")
    parser.add_argument("--disable-ocr-fallback", action="store_true", help="OCR fallback বন্ধ করতে")
    parser.add_argument("--ocr-language", default="ben+eng", help="Tesseract OCR language code")
    parser.add_argument("--ocr-oem", type=int, choices=[0, 1, 2, 3], default=1, help="Tesseract OCR Engine Mode")
    parser.add_argument("--ocr-psm", type=int, default=3, help="Tesseract Page Segmentation Mode")
    parser.add_argument("--ocr-probe-pages", type=int, default=8, help="OCR প্রয়োজন কিনা detect করার জন্য প্রথম কয়েকটি পৃষ্ঠা")
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.74,
        help="0-1 স্কেলে minimum quality threshold",
    )
    parser.add_argument("--start-chapter", type=int, default=1, help="কোন chapter index থেকে শুরু হবে (1-based)")
    parser.add_argument("--max-chapters", type=int, default=0, help="মোট কতটি chapter প্রসেস হবে (0 = সব)")
    parser.add_argument("--skip-quality-judge", action="store_true", help="Groq quality-judge step skip করতে")
    parser.add_argument("--checkpoint-path", default="qa_generation.checkpoint.json", help="resume checkpoint file path")
    parser.add_argument("--resume", action="store_true", help="আগের checkpoint থেকে resume করতে")
    parser.add_argument("--rate-limit-retries", type=int, default=5, help="429 rate-limit এ সর্বোচ্চ retry count")
    parser.add_argument("--low-token-mode", action="store_true", help="token খরচ কমাতে ছোট context ও কম candidate ব্যবহার করবে")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        build_dataset(
            pdf_path=args.pdf,
            output_csv=args.output,
            groq_model=args.model,
            embedding_model=args.embedding_model,
            questions_per_chapter=args.questions_per_chapter,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            chunk_group_size=args.chunk_group_size,
            chunk_group_stride=args.chunk_group_stride,
            qa_pairs_per_call=args.qa_pairs_per_call,
            top_k=args.top_k,
            quality_threshold=args.quality_threshold,
            min_chapter_chars=args.min_chapter_chars,
            enable_ocr_fallback=not args.disable_ocr_fallback,
            ocr_language=args.ocr_language,
            ocr_oem=args.ocr_oem,
            ocr_psm=args.ocr_psm,
            ocr_probe_pages=args.ocr_probe_pages,
            start_chapter=args.start_chapter,
            max_chapters=args.max_chapters,
            skip_quality_judge=args.skip_quality_judge,
            checkpoint_path=args.checkpoint_path,
            resume=args.resume,
            rate_limit_retries=args.rate_limit_retries,
            low_token_mode=args.low_token_mode,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"❌ ত্রুটি: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
