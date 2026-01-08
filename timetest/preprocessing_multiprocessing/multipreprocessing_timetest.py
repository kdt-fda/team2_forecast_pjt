import pandas as pd
import os
import numpy as np
import time
import sys
from ekonlpy.sentiment import MPCK
from multiprocessing import Pool
from tqdm import tqdm

# [패치] 윈도우 인코딩 에러 방지
if sys.platform == 'win32':
    import _io
    def _patched_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if 'b' not in mode and encoding is None:
            encoding = 'utf-8'
        return _io.open(file, mode, buffering, encoding, errors, newline, closefd, opener)
    import builtins
    builtins.open = _patched_open
os.environ['PYTHONUTF8'] = '1'

# 1. 전역 변수로 선언만 해둡니다.
worker_mpck = None

# 2. 일꾼들이 처음 출근했을 때 딱 한 번만 실행할 함수
def init_worker():
    global worker_mpck
    from ekonlpy.sentiment import MPCK
    if worker_mpck is None:
        worker_mpck = MPCK()

# --- [1. ngramize 함수] ---
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

# --- [2. 멀티프로세싱 일꾼 함수] ---
def worker_task(text_list):
    global worker_mpck
    batch_results = []
    for text in text_list:
        try:
            tokens = worker_mpck.tokenize(text)
            final = ngramize(tokens, max_n=5)
            batch_results.append(final)
        except Exception as e:
            print(f"❌ 분석 에러 발생: {e}")
            batch_results.append([]) 
    return batch_results

# --- [3. 메인 실행 제어기] ---
def run_production(df, output_folder='./processed_batches', batch_size=2000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_cores = 8
    total_batches = int(np.ceil(len(df) / batch_size))
    print(f"⚙️ 총 {len(df)}건을 {num_cores}개 코어로 처리합니다.")

    with Pool(num_cores, initializer=init_worker) as pool:
        total_batches = int(np.ceil(len(df) / batch_size))
        for i in tqdm(range(total_batches), desc="Processing Batches"):
            batch_file = os.path.join(output_folder, f"batch_{i}.parquet")
            if os.path.exists(batch_file): continue
            start = i * batch_size
            end = min((i + 1) * batch_size, len(df))
            chunk = df.iloc[start:end].copy()
            split_chunks = np.array_split(chunk['content'], num_cores)
            results = pool.map(worker_task, split_chunks)
            flat_results = [item for sublist in results for item in sublist]
            chunk['tokens'] = flat_results
            chunk.to_parquet(batch_file)

if __name__ == "__main__":
    
    
    SENTENCE_FILE = 'df_sentences_timetest.parquet'
    if os.path.exists(SENTENCE_FILE):
        print(f"✅ 파일을 로드합니다: {SENTENCE_FILE}")
        df_sentences = pd.read_parquet(SENTENCE_FILE)
        start_total = time.time()
        run_production(df_sentences)
        end_total = time.time()
    else:
        print("❌ 전처리된 파일이 없습니다. 1번 파일을 먼저 실행하세요.")
    print(f"✨ 전체 소요 시간: {(end_total - start_total)/60:.2f}분")