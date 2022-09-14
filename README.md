# CRS_212HO
KT Ho 

`data/redial` 과 `saved_model/` 에 현재 필요한 파일이 github에 존재하지 않는 상황
`init_git_load.sh` 라는 shell 파일로 onedrive에 공유링크를 통해 받을 수 있도록 세팅해놓은 상황입니다.

## 1. Git 폴더 clone
`git clone https://github.com/HYEONGJUN-JANG/CRS_212HO.git` 을 통해 다운로드

## 2. 폴더진입 후 dataset과 model.pt 다운로드
``` shell
cd CRS_212HO
sh init_git_load.sh
```
## 3. 필요한 dataset 다운로드 마친 이후, 가상환경 관련 세팅
```shell
conda env create --file env4ktserver.yaml 
```

## 4. main.py 를 통한 실행
```shell
python main.py --name=review --n_sample=1 --max_review_len=200 
```
위와같은 형식으로 파라미터를 제공하며 실행할 수 있습니다. 
parameters.py 파일을 확인하면 인자로 제공 가능한 파라미터 목록을 확인할 수 있습니다.

