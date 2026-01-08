import pandas as pd
import os
import kss
from tqdm import tqdm

def run_kss_step():
    tqdm.pandas()

    SENTENCE_FILE = 'df_sentences_timetest.parquet'
    
    # 1. 데이터 로드 (nrows=1500 테스트용 설정 유지)
    print("📂 원본 데이터를 로드하고 합치는 중...")
    news = pd.read_csv('../db/preprocessing/news_preprocessed_fixed.csv', encoding='utf-8', nrows=1500)
    meetings = pd.read_csv('../db/preprocessing/meeting_preprocessed_fixed.csv', encoding='utf-8', nrows=1500)
    reports = pd.read_csv('../db/preprocessing/final_integrated_full_v2.csv', encoding='utf-8', nrows=1500)
    press = pd.read_csv('../db/preprocessing/press_preprocessed_fixed.csv', encoding='utf-8', nrows=1500)

    df_total = pd.concat([news, meetings, reports, press], ignore_index=True)
    df_total['doc_id'] = df_total.index
    
    # 필요한 컬럼만 추출 및 결측치 제거
    final_cols = ['date', 'content', 'category', 'source', 'doc_id']
    df_total = df_total[final_cols].dropna(subset=['content'])

    # 2. 문장 분리 작업 (KSS)
    print("✂️ 문장 분리(KSS)를 시작합니다...")
    df_total['content'] = df_total['content'].progress_apply(kss.split_sentences)
    
    # 문장별로 행 분리(explode)
    df_sentences = df_total.explode('content').reset_index(drop=True)
    df_sentences['tokens'] = None # 이후 단계를 위해 빈 컬럼 생성

    # 3. 결과 저장
    print(f"💾 쪼개진 데이터를 {SENTENCE_FILE}로 저장합니다...")
    df_sentences.to_parquet(SENTENCE_FILE)


if __name__ == "__main__":
    run_kss_step()