import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin #상대경로 완전한 url로
import html
import os
from pathlib import Path



BASE_URL ="https://www.bok.or.kr"
#연도별 목록 경로
LIST_URL = "https://www.bok.or.kr/portal/singl/crncyPolicyDrcMtg/listYear.do"
#쿼리 파라미터 문자열
LIST_PARAMS = "mtgSe=A&menuNo=200755&pYear="
#다운로드 경로
download_path = "/portal/cmmn/file/fileDown.do?menuNo=200755&atchFileId=7591184b562d4dce19b42cd4e38059a4&fileSn=4"

#연도 페이지 URL 만들기 함수

def make_year_url(year):
    return LIST_URL + "?" + LIST_PARAMS + str(year)

#파일 다운로드 URL만들기 함수

def make_download_url():
    return BASE_URL+ download_path

def fetch_year_html(year):
    url = make_year_url(year)
    headers = {"User-agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text

def find_column_index(soup, col_name):
    ths = soup.select("table thead th")
    for i, th in enumerate(ths):
        if col_name in th.get_text(strip=True):
            return i
    return None


def extract_pdf_link(html_text):
    soup = BeautifulSoup(html_text,'lxml')
    
    col_idx=find_column_index(soup, "기자간담회")
    result = []
    rows = soup.select("table tbody tr")
    
    for tr in rows :
        tds = tr.find_all("td", recursive=False)
        body_idx = col_idx -1 #thead에는 td가 5개 그런데 table에는 td가 4개여서 수정
        if body_idx < 0 or len(tds) <= body_idx:
            continue
        press_td = tds[body_idx] #기자간담회 col만

        for i in press_td.select('a[href*="fileDown.do"]') :
            #pdf만 : title 확장자로 필터링
            title = (i.get("title") or "").strip()
            if not title.lower().endswith(".pdf"):
                continue

            href = html.unescape(i.get("href") or "").strip()
            full_url = urljoin(BASE_URL, href)

            result.append((title, full_url))

    return result

def collect_year_link(year):
    html_text = fetch_year_html(year)
    links = extract_pdf_link(html_text)
    return links


#실행부
if __name__ == "__main__":
    
    pdf_links = []

    for year in range(2012, 2026):  # 2005~2025
        links = collect_year_link(year)
        print(year, "count:", len(links))
        pdf_links.extend([(year, name, url) for (name, url) in links])

    print("TOTAL:", len(pdf_links))

    save_root = "bok_pdfs"
    os.makedirs(save_root, exist_ok=True)

    
    save_root = "bok_pdfs"
    os.makedirs(save_root, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0"}

    for year, name, url in pdf_links:
        year_dir = os.path.join(save_root, str(year))
        os.makedirs(year_dir, exist_ok=True)

        file_path = os.path.join(year_dir, name)

        # 이미 받았으면 건너뛰기
        if os.path.exists(file_path):
            continue

        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(r.content)

        print("downloaded:", file_path, "bytes:", len(r.content))
    

