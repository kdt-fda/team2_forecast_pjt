import sys
import os
import time

# [1. 필수 패치] 윈도우에서 라이브러리 내부 파일 읽기 에러 방지
if sys.platform == 'win32':
    import _io
    def _patched_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if 'b' not in mode and encoding is None:
            encoding = 'utf-8'
        return _io.open(file, mode, buffering, encoding, errors, newline, closefd, opener)
    import builtins
    builtins.open = _patched_open

# UTF-8 환경 변수 설정
os.environ['PYTHONUTF8'] = '1'

import pandas as pd
from ekonlpy.sentiment import MPCK
from tqdm import tqdm

# --- [2. ngramize 함수] ---
def ngramize(tokens, max_n=5):
    keep_tags = ['NNG', 'VA', 'VAX', 'MAG', 'VV']
    filtered = [w for w in tokens if w.split('/')[1] in keep_tags]
    all_ngrams = []
    for pos in range(len(filtered)):
        for n in range(1, max_n + 1):
            if pos + n <= len(filtered):
                ngram = ";".join(filtered[pos : pos + n])
                all_ngrams.append({'ngram': ngram, 'start': pos, 'end': pos + n, 'len': n})
                
    final_ngrams = []
    sorted_ngrams = sorted(all_ngrams, key=lambda x: x['len'], reverse=True)
    covered_ranges = set()
    for ngram in sorted_ngrams:
        is_covered = False
        for i in range(ngram['start'], ngram['end']):
            if i in covered_ranges:
                is_covered = True
                break
        if not is_covered:
            final_ngrams.append(ngram['ngram'])
            for i in range(ngram['start'], ngram['end']):
                covered_ranges.add(i)
    return final_ngrams

# --- [3. MPCK 선언 및 토큰화 함수] ---
# 패치 적용 후에 선언해야 안전합니다.
mpck = MPCK()

def get_final_tokens(text):
    if pd.isna(text) or text == "":
        return []
    try:
        basic_tokens = mpck.tokenize(text)
        return ngramize(basic_tokens, max_n=5)
    except Exception as e:
        # 에러 발생 시 로그 출력 후 빈 리스트 반환
        return []

# --- [4. 실행 로직] ---
if __name__ == "__main__":
    # 전체 성능 측정 시작
    
    tqdm.pandas()

    # KSS가 이미 완료된 파일 로드
    SENTENCE_FILE = 'df_sentences_timetest.parquet'
    
    if os.path.exists(SENTENCE_FILE):
        print(f"📂 KSS 전처리 파일을 로드합니다: {SENTENCE_FILE}")
        df_sentences = pd.read_parquet(SENTENCE_FILE)
    
        
        print(f"🧠 총 {len(df_sentences)}건의 문장에 대해 싱글 코어 분석을 시작합니다...")
        total_start = time.time()
        # 단일 프로세싱 실행 (progress_apply로 진행 바 표시)
        df_sentences['tokens'] = df_sentences['content'].progress_apply(get_final_tokens)
        total_end = time.time()
        
        # 상위 10개 결과 확인용 출력
        print("\n🔎 분석 결과 샘플:")
        print(df_sentences[['content', 'tokens']].head(10))
        
    else:
        print(f"❌ '{SENTENCE_FILE}' 파일이 없습니다. KSS 전처리 파일을 먼저 생성해주세요.")

    # 성능 측정 종료
 
    total_minutes = (total_end - total_start) / 60

    print("-" * 40)
    print(f"✨ 분석이 완료되었습니다!")
    print(f"⏱️ 싱글 프로세스 총 소요 시간: {total_minutes:.2f}분")
    print("-" * 40)