import json
import re
import ast
from collections import Counter
from pathlib import Path

import pandas as pd

IN_PATH  = Path(r"db\press_conference_pdfs\processed\docs_tokens.csv")
OUT_PATH = Path(r"db\press_conference_pdfs\processed\docs_tokens_clean_v2.csv")

TOK_COL = "tokens"  # JSON list 문자열 컬럼

# 표면형 기준 제거(연락처/도메인/웹 찌꺼기)
STOP_SURF = {
    "tel", "fax", "mail", "email", "e-mail",
    "http", "https", "www",
    "bok", "or", "kr",
    "co", "com", "net", "org",
}

# 경제/금융 약어는 남길 수도 있음(원하면 추가)
KEEP_EN_ABBR = {
    "imf", "gdp", "cpi", "ppi", "kospi", "cds", "bp", "fed", "ecb"
}

# 도메인 패턴
DOMAIN_RE = re.compile(r".+\.(kr|com|net|org|co|io|ai|go\.kr|or\.kr)$", re.IGNORECASE)
# url 시작
URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
# 이메일
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", re.IGNORECASE)

# "표면형/POS" 분리용
SURF_POS_RE = re.compile(r"^(.+?)/([A-Z]{1,5})$")

# 숫자만/기호만
NUM_ONLY_RE = re.compile(r"^\d+(\.\d+)?$")
SYMBOL_ONLY_RE = re.compile(r"^[\W_]+$")

def safe_load_tokens(s):
    """tokens가 JSON 문자열(list)이거나, 파이썬 리스트 문자열이어도 로드"""
    if not isinstance(s, str) or not s.strip():
        return []
    s = s.strip()

    # 1) JSON: ["a","b"]
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except Exception:
        pass

    # 2) Python list literal: ['a','b']
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []

def split_surface_pos(tok: str):
    """'Tel/SL' -> ('Tel','SL'), '금리/NNG' -> ('금리','NNG'), 그 외 -> (tok,None)"""
    t = str(tok).strip().strip('"').strip("'")
    m = SURF_POS_RE.match(t)
    if m:
        return m.group(1), m.group(2)
    return t, None

def is_english_word(s: str) -> bool:
    return bool(re.fullmatch(r"[a-z]+", s))

def should_drop(surface: str) -> bool:
    s = surface.strip()
    if not s:
        return True

    low = s.lower()

    # 0) 고정 제거 표면형
    if low in STOP_SURF:
        return True

    # 1) url/도메인/이메일 흔적
    if URL_RE.match(low):
        return True
    if DOMAIN_RE.match(low):
        return True
    if EMAIL_RE.match(low):
        return True
    if "@" in low:  # 조각 형태 방어
        return True

    # 2) 숫자만/기호만/너무 짧은 토큰
    if NUM_ONLY_RE.match(s):
        return True
    if SYMBOL_ONLY_RE.match(s):
        return True
    if len(s) <= 1:
        return True

    # 3) 영문 일반단어 제거(경제 약어는 KEEP)
    if is_english_word(low) and low not in KEEP_EN_ABBR:
        return True

    return False

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"입력 파일 없음: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    if TOK_COL not in df.columns:
        raise KeyError(f"'{TOK_COL}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    removed = Counter()

    def clean_cell(cell):
        toks = safe_load_tokens(cell)
        out = []

        for tok in toks:
            surf, pos = split_surface_pos(tok)
            if should_drop(surf):
                removed[surf.lower()] += 1
                continue

            # 원래 형식 유지: POS가 있으면 surf/POS로 저장
            if pos:
                out.append(f"{surf}/{pos}")
            else:
                out.append(surf)

        return json.dumps(out, ensure_ascii=False)

    before_avg = df[TOK_COL].apply(lambda x: len(safe_load_tokens(x))).mean()
    df[TOK_COL] = df[TOK_COL].apply(clean_cell)
    after_avg = df[TOK_COL].apply(lambda x: len(safe_load_tokens(x))).mean()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved: {OUT_PATH}")
    print(f"[INFO] docs: {len(df)}")
    print(f"[INFO] avg token_len: {before_avg:.1f} -> {after_avg:.1f}")
    print("[INFO] top removed surfaces:")
    for k, v in removed.most_common(30):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()