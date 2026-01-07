from pathlib import Path

PDF_ROOT = Path(r"C:\Users\gusgh\Documents\GitHub\team2_forecast_pjt\db\press_conference_pdfs")
OUT_DIR = PDF_ROOT / "processed"
DOC_COUNTS_DIR = OUT_DIR / "doc_token_counts"
TEXT_DIR = OUT_DIR / "extracted_text"

SAVE_EXTRACTED_TXT = False

KEEP_POS = {"NNG", "NNP", "NNB", "VV", "VA", "MAG", "SL", "SN"}
MIN_TOKEN_LEN = 2
DROP_NUM_ONLY = True
