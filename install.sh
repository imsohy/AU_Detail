#!/bin/bash
# AU_Detail 프로젝트 설치 스크립트
# 이 스크립트는 conda 환경을 생성하고 필요한 패키지를 설치합니다.
#
# 사용법:
#   chmod +x install.sh
#   ./install.sh
# 또는
#   bash install.sh
#
# 비대화형 모드 (기존 환경 자동 삭제):
#   ./install.sh --yes

set -e  # 오류 발생 시 스크립트 중단

# 비대화형 모드 확인
AUTO_YES=false
if [[ "$1" == "--yes" ]] || [[ "$1" == "-y" ]]; then
    AUTO_YES=true
fi

ENV_NAME="AU_Detail"
PYTHON_VERSION="3.9.18"
PYTORCH_VERSION="1.12.1"
TORCHVISION_VERSION="0.13.1"
TORCHAUDIO_VERSION="0.12.1"
CUDA_VERSION="11.3"

echo "=========================================="
echo "AU_Detail 프로젝트 설치를 시작합니다"
echo "=========================================="

# conda가 설치되어 있는지 확인
if ! command -v conda &> /dev/null; then
    echo "오류: conda가 설치되어 있지 않습니다."
    echo "Anaconda 또는 Miniconda를 먼저 설치해주세요."
    exit 1
fi

# conda 초기화 확인 및 설정
echo "conda 초기화 확인 중..."
# conda base 경로 확인
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ]; then
    echo "오류: conda base 경로를 찾을 수 없습니다."
    exit 1
fi

# conda shell hook 초기화
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    # conda.sh가 없는 경우 eval 사용
    eval "$(conda shell.bash hook)"
fi

# 기존 환경이 있는지 확인
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "경고: '${ENV_NAME}' 환경이 이미 존재합니다."
    if [ "$AUTO_YES" = true ]; then
        echo "비대화형 모드: 기존 환경을 자동으로 삭제합니다."
        conda env remove -n ${ENV_NAME} -y
    else
        read -p "기존 환경을 삭제하고 새로 만들까요? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "기존 환경 삭제 중..."
            conda env remove -n ${ENV_NAME} -y
        else
            echo "설치를 취소합니다."
            exit 1
        fi
    fi
fi

# 1) 새 환경 생성
echo ""
echo "1단계: conda 환경 생성 중..."

# conda 캐시 정리 (선택적, 오류 발생 시 자동 실행)
if ! conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y 2>/dev/null; then
    echo "conda 환경 생성 중 오류 발생. 캐시를 정리하고 재시도합니다..."
    conda clean --all -y
    # 플러그인 없이 재시도
    CONDA_NO_PLUGINS=true conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y || {
        echo "오류: conda 환경 생성에 실패했습니다."
        echo "다음 명령어를 수동으로 실행해보세요:"
        echo "  conda clean --all -y"
        echo "  conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y"
        exit 1
    }
fi

# 2) 환경 활성화
echo ""
echo "2단계: 환경 활성화 중..."
conda activate ${ENV_NAME}

# 환경이 제대로 활성화되었는지 확인
if [ "$CONDA_DEFAULT_ENV" != "${ENV_NAME}" ]; then
    echo "오류: 환경 활성화에 실패했습니다."
    exit 1
fi
echo "환경 활성화 완료: ${CONDA_DEFAULT_ENV}"

# 3) pip 업그레이드
echo ""
echo "3단계: pip 업그레이드 중..."
python -m pip install --upgrade pip

# 4) PyTorch + CUDA 11.3 설치
echo ""
echo "4단계: PyTorch 및 CUDA 툴킷 설치 중..."
if ! conda install pytorch=${PYTORCH_VERSION} torchvision=${TORCHVISION_VERSION} torchaudio=${TORCHAUDIO_VERSION} cudatoolkit=${CUDA_VERSION} -c pytorch -c nvidia -y 2>/dev/null; then
    echo "conda install 중 오류 발생. 플러그인 없이 재시도합니다..."
    CONDA_NO_PLUGINS=true conda install pytorch=${PYTORCH_VERSION} torchvision=${TORCHVISION_VERSION} torchaudio=${TORCHAUDIO_VERSION} cudatoolkit=${CUDA_VERSION} -c pytorch -c nvidia -y || {
        echo "오류: PyTorch 설치에 실패했습니다."
        exit 1
    }
fi

# 5) requirements.txt 설치 (PyTorch 관련 패키지는 제외)
echo ""
echo "5단계: requirements.txt 패키지 설치 중..."
echo "주의: 이 과정은 시간이 걸릴 수 있습니다..."

# requirements.txt에서 PyTorch 관련 패키지를 제외한 임시 파일 생성
TEMP_REQUIREMENTS=$(mktemp)
grep -v "^torch==" requirements.txt | grep -v "^torchvision==" | grep -v "^torchaudio==" > ${TEMP_REQUIREMENTS}

# PyTorch를 제외한 패키지 설치
pip install -r ${TEMP_REQUIREMENTS}

# 임시 파일 삭제
rm ${TEMP_REQUIREMENTS}

echo "PyTorch는 이미 conda를 통해 설치되었습니다."

echo ""
echo "=========================================="
echo "설치가 완료되었습니다!"
echo "=========================================="
echo ""
echo "환경을 활성화하려면 다음 명령어를 실행하세요:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "환경을 비활성화하려면:"
echo "  conda deactivate"
echo ""