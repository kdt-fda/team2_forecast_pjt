import pandas as pd
import os
import numpy as np
from ekonlpy.sentiment import MPCK
from multiprocessing import Pool, cpu_count

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

# --- [2. 멀티프로세싱용 개별 일꾼(Worker) 함수] ---
def worker_task(text_list):
    mpck = MPCK() 
    batch_results = []
    for text in text_list:
        try:
            tokens = mpck.tokenize(text)
            final = ngramize(tokens, max_n=5)
            batch_results.append(final)
        except Exception as e:
            print(f"에러 발견: {e}")
            raise e
    return batch_results

# --- [3. 메인 실행 제어기] ---
def run_production(df, output_folder='./processed_batches', batch_size=2000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_cores = 8
    total_batches = int(np.ceil(len(df) / batch_size))
    
    print(f"⚙️ 총 {len(df)}건 데이터를 {num_cores}개 코어로 처리합니다.")

    for i in tqdm(range(total_batches), desc="Processing Batches"):
        batch_file = os.path.join(output_folder, f"batch_{i}.parquet")
        
        # [체크포인트] 이미 처리된 파일은 건너뛰기! (맘 편하게 재실행 가능)
        if os.path.exists(batch_file):
            continue
            
        start = i * batch_size
        end = min((i + 1) * batch_size, len(df))
        chunk = df.iloc[start:end].copy()
        
        # 병렬 처리 시작
        with Pool(num_cores) as pool:
            # 데이터를 코어 개수만큼 다시 쪼개서 분배
            split_chunks = np.array_split(chunk['content'], num_cores)
            results = pool.map(worker_task, split_chunks)
            
            # 쪼개진 결과 합쳐서 컬럼에 넣기
            flat_results = [item for sublist in results for item in sublist]
            chunk['tokens'] = flat_results
            
            # Parquet 형식으로 저장 (csv보다 빠르고 용량이 작음)
            chunk.to_parquet(batch_file)

# --- [4. 전체 실행 로직] ---
if __name__ == "__main__":
    import kss
    from tqdm import tqdm
    tqdm.pandas()

    SENTENCE_FILE = 'df_sentences.parquet'
    if os.path.exists(SENTENCE_FILE):
        print(f"✅ 이미 쪼개진 파일({SENTENCE_FILE})을 찾았습니다. 로드 중...")
        df_sentences = pd.read_parquet(SENTENCE_FILE)
    else:
        # 1. 데이터 로드 (파일이 없을 때만 원본 CSV들을 읽어옵니다)
        print("📂 원본 데이터를 로드하고 합치는 중...")
        news = pd.read_csv('../db/preprocessing/news_preprocessed_fixed.csv', encoding='utf-8')
        meetings = pd.read_csv('../db/preprocessing/meeting_preprocessed_fixed.csv', encoding='utf-8')
        reports = pd.read_csv('../db/preprocessing/final_integrated_full_v2.csv', encoding='utf-8')
        press = pd.read_csv('../db/preprocessing/press_preprocessed_fixed.csv', encoding='utf-8')

        df_total = pd.concat([news, meetings, reports, press], ignore_index=True)
        # 3. 문서 고유 Index
        df_total['doc_id'] = df_total.index
        final_cols = ['date', 'content', 'tokens', 'category', 'source', 'doc_id']
        df_total = df_total[final_cols]
        df_total = df_total.dropna(subset=['content'])

        # 2. 문장 분리 작업 (KSS는 여기서 미리 수행)
        print("✂️ 문장 분리(KSS)를 시작합니다...")
        df_working = df_total.copy()
        df_working['content'] = df_working['content'].progress_apply(kss.split_sentences)
        df_sentences = df_working.explode('content').reset_index(drop=True)

        del df_total
        del df_working

        df_sentences['tokens'] = None
        output_columns = ['doc_id', 'date', 'content', 'tokens', 'category', 'source']
        df_sentences = df_sentences[output_columns]
        
        print(f"💾 쪼개진 데이터를 {SENTENCE_FILE}로 저장합니다...")
        df_sentences.to_parquet(SENTENCE_FILE)

    # 3. 멀티프로세싱 실행
    run_production(df_sentences)
    
    print("✨ 모든 작업이 완료되었습니다! './processed_batches' 폴더를 확인하세요.")