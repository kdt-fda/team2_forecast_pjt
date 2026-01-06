import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import preprocess_config as cfg

# 필요하면 여기에 추가
STOPWORDS = {
    "tel", "fax", "mail", "email", "e-mail",
    "http", "https", "www",
    "bok", "or", "kr",
    "co", "com", "net", "org",
}


# 영어 토큰 전부 제거하고 싶으면 True (권장: 일단 False로 시작)
DROP_ALL_ENGLISH = False

def safe_load_tokens(s: str):
    try:
        return json.loads(s) if isinstance(s, str) and s.strip() else []
    except Exception:
        return []
    
def refine_tokens(tokens):
    removed = []
    kept = []

    for t in tokens:
        tok = str(t).strip()
        if not tok :
            continue

        low = tok.lower()

        # 1) 고정 불용어
        if low in STOPWORDS:
            removed.append(low)
            continue

        # 2) URL/도메인/이메일 흔적
        if low.startswith("http"):
            removed.append(low)
            continue

        if '@' in low :
            removed.append(low)
            continue
        if re.search(r"\b(www|http|https)\b", low):
            removed.append(low)
            continue
        if re.search(r"\.(kr|com|net|org)\b", low):
            removed.append(low)
            continue
        # 3) 영문 전부 제거 옵션
        if DROP_ALL_ENGLISH and re.fullmatch(r"[a-z]+", low):
            removed.append(low)
            continue

        kept.append(tok)

    return kept, removed


def main():
    in_path = Path(cfg.OUT_DIR) / "docs_tokens.csv"
    out_path = Path(cfg.OUT_DIR) / "docs_tokens_clean.csv"

    if not in_path.exists():
        raise FileNotFoundError(f"입력 CSV 없음: {in_path}")

    df = pd.read_csv(in_path)

    required = ["date", "content", "tokens", "category", "source"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"컬럼 누락: {missing}")

    removed_counter = Counter()
    before_lens = []
    after_lens = []

    new_tokens_json = []
    for s in df["tokens"].tolist():
        tokens = safe_load_tokens(s)
        before_lens.append(len(tokens))

        kept, removed = refine_tokens(tokens)
        after_lens.append(len(kept))
        removed_counter.update(removed)

        new_tokens_json.append(json.dumps(kept, ensure_ascii=False))

    df["tokens"] = new_tokens_json
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved: {out_path}")
    print(f"[INFO] docs: {len(df)}")
    print(f"[INFO] avg token_len: {sum(before_lens)/len(before_lens):.1f} -> {sum(after_lens)/len(after_lens):.1f}")

    print("[INFO] top removed tokens:")
    for tok, cnt in removed_counter.most_common(20):
        print(f"  {tok}: {cnt}")


if __name__ == "__main__":
    main()    