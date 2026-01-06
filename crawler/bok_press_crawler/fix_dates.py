import re
from pathlib import Path

import pandas as pd
import pdfplumber
import preprocess_config as cfg


IN_PATH = Path(cfg.OUT_DIR) / "docs_tokens_clean.csv"
OUT_PATH = Path(cfg.OUT_DIR) / "docs_tokens_clean_dated.csv"


def normalize_date(y, m, d):
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def extract_date_from_text(text: str) -> str:
    if not text:
        return ""

    head = text[:4000]  # 앞부분에서만 찾기(속도/오탐 방지)

    # 1) 2012. 1. 12 / 2012-1-12 / 2012/01/12 (구분자 다양)
    m = re.search(r"((?:19|20)\d{2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})", head)
    if m:
        return normalize_date(m.group(1), m.group(2), m.group(3))

    # 2) 2012년 1월 12일
    m = re.search(r"((?:19|20)\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", head)
    if m:
        return normalize_date(m.group(1), m.group(2), m.group(3))

    return ""


def extract_date_from_pdf(pdf_path: Path) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            texts = []
            for i in range(min(2, len(pdf.pages))):  # 0~1페이지
                t = pdf.pages[i].extract_text() or ""
                if t.strip():
                    texts.append(t)
            return extract_date_from_text("\n".join(texts))
    except Exception:
        return ""


def fallback_date_from_path(pdf_path: Path) -> str:
    # 폴더명에서 연도
    year = None
    for part in pdf_path.parts[::-1]:
        if re.fullmatch(r"(19|20)\d{2}", part):
            year = int(part)
            break

    # 파일명 (1201) -> 월만 추정
    m = re.search(r"\((\d{4})\)", pdf_path.name)
    if year and m:
        code = m.group(1)
        month = int(code[2:4])
        return normalize_date(year, month, 1)

    return ""


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"입력 CSV 없음: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    pdf_paths = sorted(Path(cfg.PDF_ROOT).rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"PDF 없음: {cfg.PDF_ROOT}")

    # 전처리 때와 동일하게 "PDF 1개 = CSV 1행" 순서로 가정
    n = min(len(pdf_paths), len(df))
    df = df.iloc[:n].copy()
    paths = pdf_paths[:n]

    df["filename"] = [p.name for p in paths]
    df["rel_path"] = [str(p.relative_to(Path(cfg.PDF_ROOT))) for p in paths]

    dates = []
    miss = 0

    for p in paths:
        d = extract_date_from_pdf(p)
        if not d:
            d = fallback_date_from_path(p)
        if not d:
            miss += 1
            d = ""
        dates.append(d)

    df["date"] = dates
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved: {OUT_PATH}")
    print(f"[INFO] rows: {len(df)}, empty_date: {miss}")
    print("[INFO] date sample:", df['date'].head(12).to_list())


if __name__ == "__main__":
    main()
