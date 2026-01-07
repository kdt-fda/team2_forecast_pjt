import re
from pathlib import Path
import pdfplumber

def tokenize(text, pos_fn, keep_pos, min_len=2, drop_num_only=True, include_pos=False):
    tokens = []
    for w, p in pos_fn(text):
        w = w.strip()
        if not w:
            continue
        if p not in keep_pos:
            continue
        if len(w) < min_len:
            continue
        if drop_num_only and w.isdigit():
            continue

        tokens.append(f"{w}/{p}" if include_pos else w)
    return tokens

def load_tagger():
    """
    Mecab(eKoNLPy) 우선, 실패 시 Okt로 폴백.
    반환: (tagger_name, pos_fn)
    pos_fn(text) -> List[(token, pos)]
    """
    try:
        from ekonlpy.tag import Mecab
        tagger = Mecab()

        def pos_fn(text: str):
            return tagger.pos(text)

        return "Mecab(eKonlpy)", pos_fn

    except Exception:
        from konlpy.tag import Okt
        okt = Okt()

        def pos_fn(text: str):
            return okt.pos(text, norm=True, stem=True)

        return "Okt(konlpy)", pos_fn


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    """PDF 전체 페이지 텍스트 추출. (text, page_count) 반환"""
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t)
    return "\n".join(texts), page_count


def clean_text(text: str) -> str:
    """텍스트 정규화(기본). 필요하면 규칙 추가."""
    text = text.replace("\u00a0", " ")  # NBSP
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 한글/영문/숫자/기본문장부호 정도만 남김
    text = re.sub(r"[^0-9A-Za-z가-힣\.\,\%\(\)\-\n ]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def parse_date_from_name(name: str) -> str:
    """파일명/폴더명에서 날짜 파싱. 실패하면 ''"""
    # YYYYMMDD
    m = re.search(r"(19|20)\d{2}[01]\d[0-3]\d", name)
    if m:
        s = m.group(0)
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"

    # YYYY-MM-DD / YYYY.MM.DD / YYYY_MM_DD
    m = re.search(r"(19|20)\d{2}[-._][01]\d[-._][0-3]\d", name)
    if m:
        s = m.group(0).replace(".", "-").replace("_", "-")
        return s

    return ""


def infer_category_source(pdf_path: Path) -> tuple[str, str]:
    """폴더/파일명 기반 휴리스틱. 필요하면 너 폴더명에 맞게 수정."""
    p = str(pdf_path).lower()

    category = "unknown"
    if "minutes" in p or "회의록" in p or "mpb" in p:
        category = "minutes"
    elif "press" in p or "기자" in p or "간담회" in p:
        category = "press"
    elif "news" in p or "기사" in p:
        category = "news"
    elif "report" in p or "리포트" in p:
        category = "report"

    source = "BOK"
    if "infomax" in p:
        source = "Infomax"
    elif "naver" in p:
        source = "Naver"

    return category, source


def filter_tokens(pos_list, keep_pos: set[str], min_len: int, drop_num_only: bool):
    """품사/길이/숫자 필터 적용"""
    filtered = []
    for token, pos in pos_list:
        token = str(token).strip()
        if not token:
            continue
        if pos not in keep_pos:
            continue
        if len(token) < min_len:
            continue
        if drop_num_only and token.isdigit():
            continue
        filtered.append((token, pos))
    return filtered

def make_tokens_from_pos(pos_list, include_pos: bool = False):
    """
    pos_list: [(token, pos), ...]
    include_pos=False: ["중구", "본점", ...]
    include_pos=True : ["중구/NNG", "본점/NNG", ...]
    """
    if not pos_list:
        return []

    if include_pos:
        return [f"{t}/{p}" for (t, p) in pos_list]
    return [t for (t, _p) in pos_list]