# 설치 가이드

본 프로그램은 다른 파이썬 패키지를 사용하며, 가상환경을 구성해 패키지를 설치할 필요성이 있다.

## 설치 방법

### 방법 1: 자동 설치 스크립트 사용 (권장)

가장 간단한 방법은 제공된 설치 스크립트를 사용하는 것입니다.

**전제 조건:**
- Anaconda 또는 Miniconda가 설치되어 있어야 합니다.

**설치 절차:**

1. 프로젝트 루트 디렉토리로 이동합니다.
   ```bash
   cd /path/to/AU_Detail
   ```

2. 설치 스크립트를 실행합니다.
   ```bash
   # 실행 권한 부여 (최초 1회만)
   chmod +x install.sh
   
   # 스크립트 실행
   ./install.sh
   ```
   
   또는 비대화형 모드로 실행 (기존 환경 자동 삭제):
   ```bash
   ./install.sh --yes
   ```

3. 설치가 완료되면 환경을 활성화합니다.
   ```bash
   conda activate AU_Detail
   ```

스크립트는 다음 작업을 자동으로 수행합니다:
- Python 3.9.18이 포함된 conda 환경 생성
- pip 업그레이드
- PyTorch 1.12.1 + CUDA 11.3 설치
- requirements.txt의 모든 패키지 설치

### 방법 2: 수동 설치 (venv 사용)

conda를 사용하지 않는 경우 venv를 사용할 수 있습니다.

1. Python (3.9.18)을 설치한다.

2. 가상환경을 생성하고 활성화한다.
   ```bash
   # 가상환경 생성
   python3.9 -m venv venv
   
   # 가상환경 활성화
   source venv/bin/activate
   ```

3. 리눅스 터미널에서 다음 명령어를 통해 파이썬 패키지 관리자(pip)를 업그레이드한다.
   ```bash
   python -m pip install --upgrade pip
   ```
   
   참고: pip는 Python과 함께 기본적으로 설치되어 있지만, 최신 버전으로 업그레이드하는 것이 권장된다.

4. `pip install -r requirements.txt` 명령어를 통해 프로젝트에 필요한 모든 패키지를 설치한다.
   ```bash
   pip install -r requirements.txt
   ```
   
   이 명령어는 `requirements.txt` 파일에 명시된 모든 패키지와 버전을 자동으로 설치한다.
   
   **주의:** venv를 사용하는 경우 PyTorch와 CUDA는 별도로 설치해야 합니다.
   ```bash
   pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
   ```

## 가상환경 비활성화

작업이 끝난 후 가상환경을 비활성화하려면 다음 명령어를 실행한다:

**conda 환경인 경우:**
```bash
conda deactivate
```

**venv 환경인 경우:**
```bash
deactivate
```

## 주의사항

- 가상환경을 활성화한 상태에서만 프로그램을 실행해야 한다.
- 새로운 터미널 세션을 열 때마다 가상환경을 다시 활성화해야 한다.
- CUDA를 사용하는 경우, CUDA 및 cuDNN이 설치되어 있어야 한다.
- 설치 스크립트(`install.sh`)를 사용하는 경우, conda가 설치되어 있어야 한다.

## 문제 해결

### 설치 스크립트가 작동하지 않는 경우

1. **conda가 인식되지 않는 경우:**
   ```bash
   # conda 초기화
   conda init bash
   # 새 터미널을 열거나
   source ~/.bashrc
   ```

2. **권한 오류가 발생하는 경우:**
   ```bash
   chmod +x install.sh
   ```

3. **기존 환경과 충돌하는 경우:**
   ```bash
   # 기존 환경 삭제 후 재설치
   conda env remove -n AU_Detail
   ./install.sh --yes
   ```

4. **패키지 설치 오류가 발생하는 경우:**
   - 인터넷 연결을 확인하세요.
   - pip를 최신 버전으로 업그레이드하세요: `pip install --upgrade pip`
   - 특정 패키지 설치 오류는 해당 패키지의 공식 문서를 참조하세요.

