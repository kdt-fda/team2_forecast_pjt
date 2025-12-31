import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import preprocess_config as cfg
import preprocess_utils as ut
print("[DEBUG] preprocess_utils file:", ut.__file__)
print("[DEBUG] has load_tagger:", hasattr(ut, "load_tagger"))

import preprocess_config as cfg
print("[DEBUG] preprocess_config file:", cfg.__file__)
print("[DEBUG] attrs:", [a for a in dir(cfg) if a.isupper()])


def main():
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.DOC_COUNTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TEXT_DIR.mkdir(parents=True, exist_ok=True)

    tagger_name, pos_fn = ut.load_tagger()
    print(f"[INFO] Tagger: {tagger_name}")
    print(f"[INFO] PDF_ROOT: {cfg.PDF_ROOT}")

    pdf_files = sorted(cfg.PDF_ROOT.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"PDF를 찾지 못했습니다: {cfg.PDF_ROOT}")

    docs_tokens_rows = []   # date, content, tokens, category, source
    all_count_rows = []     # doc_id, filename, token, pos, count

    for doc_id, pdf_path in enumerate(tqdm(pdf_files, desc="Preprocess Tokens"), start=1):
        try:
            raw_text, _page_count = ut.extract_text_from_pdf(pdf_path)
            text = ut.clean_text(raw_text)
            if not text:
                continue

            if cfg.SAVE_EXTRACTED_TXT:
                (cfg.TEXT_DIR / f"{pdf_path.stem}.txt").write_text(text, encoding="utf-8")

            date = ut.parse_date_from_name(pdf_path.name) or ut.parse_date_from_name(str(pdf_path.parent))
            category, source = ut.infer_category_source(pdf_path)

            pos_list = pos_fn(text)
            pos_list = ut.filter_tokens(
                pos_list,
                keep_pos=cfg.KEEP_POS,
                min_len=cfg.MIN_TOKEN_LEN,
                drop_num_only=cfg.DROP_NUM_ONLY
            )
            if not pos_list:
                continue

            # (1) docs_tokens.csv용: tokens는 JSON 문자열로 저장
            tokens_only = [t for (t, _p) in pos_list]
            docs_tokens_rows.append({
                "date": date,
                "content": text,
                "tokens": json.dumps(tokens_only, ensure_ascii=False),
                "category": category,
                "source": source
            })

            # (2) doc_token_counts/*.csv + all_docs_token_counts.csv용: 빈도 집계
            df = pd.DataFrame(pos_list, columns=["token", "pos"])
            counts = (
                df.value_counts(["token", "pos"])
                  .reset_index(name="count")
                  .sort_values("count", ascending=False)
            )

            # 문서별 저장
            out_doc_csv = cfg.DOC_COUNTS_DIR / f"{pdf_path.stem}.csv"
            counts.to_csv(out_doc_csv, index=False, encoding="utf-8-sig")

            # 전체 통합 저장
            for row in counts.itertuples(index=False):
                all_count_rows.append({
                    "doc_id": doc_id,
                    "filename": pdf_path.name,
                    "token": row.token,
                    "pos": row.pos,
                    "count": int(row.count),
                })

        except Exception as e:
            print(f"[WARN] 실패: {pdf_path.name} / {e}")

    # 저장: docs_tokens.csv (요청한 5컬럼)
    if docs_tokens_rows:
        out_docs_tokens = cfg.OUT_DIR / "docs_tokens.csv"
        pd.DataFrame(docs_tokens_rows).to_csv(out_docs_tokens, index=False, encoding="utf-8-sig")
        print(f"[DONE] Saved: {out_docs_tokens}")

    # 저장: all_docs_token_counts.csv
    if all_count_rows:
        out_all_counts = cfg.OUT_DIR / "all_docs_token_counts.csv"
        pd.DataFrame(all_count_rows).to_csv(out_all_counts, index=False, encoding="utf-8-sig")
        print(f"[DONE] Saved: {out_all_counts}")
        print(f"[DONE] Per-doc dir: {cfg.DOC_COUNTS_DIR}")

    if not docs_tokens_rows and not all_count_rows:
        print("[DONE] 저장할 결과가 없습니다(텍스트 추출 실패 또는 토큰이 전부 필터링됨).")


if __name__ == "__main__":
    main()
