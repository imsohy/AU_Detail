## 환경 (Environment    

- OS: Linux (64-bit, Ubuntu 20.04.1)
- Python: 3.9.18
- CUDA: 11.3
- PyTorch: 1.12.1 (CUDA 11.3)
- torchvision: 0.13.1
- torchaudio: 0.12.1

추가 Python 패키지는 `requirements.txt` 참고.

## 설치 (Install)

### 방법 1: AU_Detail 환경 (conda 사용)
1. Anaconda 설치
2. 프로젝트가 깔린 루트 폴더로 cd
3. 새 환경 생성

```bash
conda create -n AU_Detail python=3.9.18
conda activate AU_Detail
```

4. PyTorch + CUDA 11.3 설치

```bash
conda install pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia
```

5. requirements.txt 설치

```bash
pip install -r requirements.txt
```

### 방법 2: deca2_cuda 환경 (pip 사용, 권장)
1. 새 환경 생성

```bash
conda create -n deca2_cuda python=3.9.18
conda activate deca2_cuda
```

2. PyTorch + CUDA 11.3 설치 (pip 사용)

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

3. CUDA 지원 확인

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

4. requirements.txt 설치

```bash
pip install -r requirements.txt
```

**참고**: `deca2_cuda` 환경은 이미 설정되어 있으며 CUDA를 지원합니다 (PyTorch 1.12.1+cu113).

## 포함되지 않은 대용량 파일

./data

./decalib/models/OpenGraphAU/checkpoints/

추후 받을 수 있는 저장소를 얻어 업로드 예정

## Train

## Demo
