# nip
2019년 2학기 소프트웨어 종합설계. NiP은 Namgyu is Pig

### dataset
Yelp 2014  
(http://www.thunlp.org/~chm/data/data.zip)  
nip/data 폴더 만들고 그 안에 dataset 저장

### preprocessing  
1. dataset 변환 (기존 .txt 파일을 리뷰 + 평점만 남긴 .txt 파일로 변환)
    ```shell script
    python3 convert.py
    ```
2. vocab 파일 생성
    ```shell script
    python3 vocab_generate.py
    ```
3. 리뷰 텍스트 토큰화 + int 값으로 인코딩
    ```shell script
    python3 encode.py
    ```
4. dataloader 생성
    - dataload.py 파일 import
    - get_loader 함수 사용 (각각 train.txt, dev.txt, test.txt를 parameter로)
    - 받은 data loader 사용
