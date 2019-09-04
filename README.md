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

2, 3번 과정에서 stanford nlp tokenizer를 다운 받고 서버를 연 상태에서 진행해야 함.  
Tokenizer 설치 및 사용법은 다음 링크 참고.  
<https://stackoverflow.com/questions/47624742/how-to-use-stanford-word-tokenizer-in-nltk>  

만일 사용이 어려울 경우 preprocessing 과정을 거친 데이터셋을 구글 드라이브에 올려놓았으니 참고.  
data 폴더 채로 받아서 사용하면 됨.
