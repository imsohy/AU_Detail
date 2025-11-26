## 환경 (Environment)

- OS: Linux (64-bit, Ubuntu 20.04.1)
- Python: 3.9.18
- CUDA: 11.3
- PyTorch: 1.12.1 (CUDA 11.3)
- torchvision: 0.13.1
- torchaudio: 0.12.1

추가 Python 패키지는 `requirements.txt` 참고.

## 설치 (Install)
1. Anaconda 설치
2. 프로젝트가 깔린 루트 폴더로 cd
3. 새 환경 생성 (이름은 원하는 걸로 바꿔도 됨)

conda create -n AU_Detail python=3.9.18
4. 환경 활성화

conda activate AU_Detail

5. PyTorch + CUDA 11.3 설치

conda install pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia

6. install requirements.txt

pip install -r requirements.txt

## 포함되지 않은 대용량 파일

./data

./decalib/models/OpenGraphAU/checkpoints/

추후 받을 수 있는 저장소를 얻어 업로드 예정

## Train

## Demo
