# 디테일 렌더링 개선 방법 사용 가이드

이 디렉토리에는 주름이 알베도 적용 시 사라지거나 옅어지는 문제를 해결하기 위한 3가지 방법이 구현되어 있습니다.

## 파일 구조

- `detail_rendering_methods.py`: 3가지 방법의 구현 코드
- `example_usage_detail_methods.py`: 사용 예제 코드
- `README_detail_methods.md`: 이 문서

## 방법 개요

### 방법 1-A: 알베도에 디테일 정보 추가
주름 영역의 알베도를 변조하여 주름이 더 잘 보이도록 합니다.

### 방법 2-A: 셰이딩 강도 조절
주름 영역의 셰이딩 대비를 강화하여 주름이 더 뚜렷하게 보이도록 합니다.

### 방법 3-A: 하이브리드 렌더링
주름은 주로 셰이딩으로, 알베도는 보조로 사용합니다.

## 사용 방법

### 1. Import 추가

`trainer_Video_DetailNewBranch_decodertrain_v3.py` 파일 상단에 다음을 추가:

```python
from detail_rendering_methods import (
    method_1A_albedo_modulation,
    method_2A_shading_enhancement,
    method_3A_hybrid_rendering
)
```

### 2. 기존 코드 수정

`trainer_Video_DetailNewBranch_decodertrain_v3.py`의 `training_step` 메서드에서:

**기존 코드 (Line 450):**
```python
uv_texture = albedo.detach() * uv_shading
```

**방법 1-A로 변경:**
```python
uv_texture = method_1A_albedo_modulation(
    albedo=albedo,
    uv_z=uv_z,
    uv_detail_normals=uv_detail_normals,
    uv_shading=uv_shading
)
```

**방법 2-A로 변경:**
```python
uv_texture = method_2A_shading_enhancement(
    albedo=albedo,
    uv_z=uv_z,
    uv_detail_normals=uv_detail_normals,
    uv_shading=uv_shading,
    enhancement_type='contrast'  # 'contrast', 'boost', 'light' 중 선택
)
```

**방법 3-A로 변경:**
```python
uv_texture = method_3A_hybrid_rendering(
    albedo=albedo,
    uv_z=uv_z,
    uv_detail_normals=uv_detail_normals,
    uv_shading=uv_shading,
    blend_mode='weighted'  # 'weighted', 'separate', 'adaptive' 중 선택
)
```

## 각 방법 상세 설명

### 방법 1-A: `method_1A_albedo_modulation`

**원리:**
- 디스플레이스먼트 맵의 크기를 계산하여 주름 영역을 감지
- 주름이 깊은 곳(디스플레이스먼트가 큰 곳)은 알베도를 약간 어둡게 변조
- 알베도 변조 범위: 0.85~1.0 (주름이 깊을수록 더 어둡게)

**파라미터:**
- `albedo`: 기본 알베도 텍스처 [B, 3, H, W]
- `uv_z`: 디스플레이스먼트 맵 [B, 1, H, W] 또는 [B, 3, H, W]
- `uv_detail_normals`: 디테일 노말맵 [B, 3, H, W]
- `uv_shading`: 셰이딩 맵 [B, 3, H, W]

**장점:**
- 구현이 간단하고 즉시 효과 확인 가능
- 주름의 색상 변화를 반영

**단점:**
- 알베도 변조 강도 조절 필요 (코드 내부의 0.15 값 조절 가능)

### 방법 2-A: `method_2A_shading_enhancement`

**원리:**
- 디스플레이스먼트 맵을 기반으로 주름 영역 감지
- 주름 영역에서 셰이딩 대비를 강화

**파라미터:**
- `albedo`: 기본 알베도 텍스처 [B, 3, H, W]
- `uv_z`: 디스플레이스먼트 맵 [B, 1, H, W] 또는 [B, 3, H, W]
- `uv_detail_normals`: 디테일 노말맵 [B, 3, H, W]
- `uv_shading`: 기본 셰이딩 맵 [B, 3, H, W]
- `enhancement_type`: 강화 방법
  - `'contrast'`: 대비 확장 방식 (기본값, 추천)
  - `'boost'`: 직접 증폭 방식
  - `'light'`: 라이트 계수 조절 방식 (lightcode, render 필요)
- `lightcode`: 라이트 계수 (enhancement_type='light'일 때만 필요)
- `render`: 렌더러 객체 (enhancement_type='light'일 때만 필요)

**장점:**
- 구현이 간단하고 효과적
- 주름 대비 개선에 효과적

**단점:**
- 과도한 증폭 시 부자연스러울 수 있음
- 하이퍼파라미터 튜닝 필요

**사용 예시:**
```python
# 대비 확장 방식 (추천)
uv_texture = method_2A_shading_enhancement(
    albedo, uv_z, uv_detail_normals, uv_shading,
    enhancement_type='contrast'
)

# 직접 증폭 방식
uv_texture = method_2A_shading_enhancement(
    albedo, uv_z, uv_detail_normals, uv_shading,
    enhancement_type='boost'
)

# 라이트 계수 조절 방식
uv_texture = method_2A_shading_enhancement(
    albedo, uv_z, uv_detail_normals, uv_shading,
    lightcode=lightcode,
    render=self.mymodel.render,
    enhancement_type='light'
)
```

### 방법 3-A: `method_3A_hybrid_rendering`

**원리:**
- 주름 영역과 평탄한 영역을 구분하여 다른 렌더링 방식 적용
- 주름 영역: 셰이딩 비중 높음
- 평탄한 영역: 알베도 비중 높음

**파라미터:**
- `albedo`: 기본 알베도 텍스처 [B, 3, H, W]
- `uv_z`: 디스플레이스먼트 맵 [B, 1, H, W] 또는 [B, 3, H, W]
- `uv_detail_normals`: 디테일 노말맵 [B, 3, H, W]
- `uv_shading`: 셰이딩 맵 [B, 3, H, W]
- `blend_mode`: 블렌딩 모드
  - `'weighted'`: 가중 합 방식 (기본값, 추천)
  - `'separate'`: 분리된 렌더링 후 블렌딩
  - `'adaptive'`: 셰이딩 강도에 따라 알베도 영향도 조절
- `alpha_min`: 평탄한 영역에서의 셰이딩 비중 (기본값: 0.3)
- `alpha_max`: 주름 영역에서의 셰이딩 비중 (기본값: 0.8)

**장점:**
- 주름과 평탄한 영역에 차별화된 처리
- 자연스러운 결과

**단점:**
- 가중치 튜닝 필요
- 구현 복잡도가 약간 높음

**사용 예시:**
```python
# 가중 합 방식 (추천)
uv_texture = method_3A_hybrid_rendering(
    albedo, uv_z, uv_detail_normals, uv_shading,
    blend_mode='weighted',
    alpha_min=0.3,  # 평탄한 영역 셰이딩 비중
    alpha_max=0.8   # 주름 영역 셰이딩 비중
)

# 분리된 렌더링 방식
uv_texture = method_3A_hybrid_rendering(
    albedo, uv_z, uv_detail_normals, uv_shading,
    blend_mode='separate'
)

# 적응형 방식
uv_texture = method_3A_hybrid_rendering(
    albedo, uv_z, uv_detail_normals, uv_shading,
    blend_mode='adaptive'
)
```

## 권장 사용 순서

1. **먼저 시도**: 방법 2-A (`enhancement_type='contrast'`)
   - 구현이 가장 간단하고 즉시 효과 확인 가능

2. **추가 개선**: 방법 1-A
   - 셰이딩만으로 부족할 때 알베도 변조 추가

3. **고급**: 방법 3-A (`blend_mode='weighted'`)
   - 더 자연스러운 결과가 필요할 때

## 하이퍼파라미터 조절

각 방법의 효과를 조절하려면 `detail_rendering_methods.py` 파일에서 다음 값들을 수정할 수 있습니다:

### 방법 1-A
- `0.15`: 알베도 변조 강도 (Line 47)
  - 값이 클수록 주름이 더 어둡게 보임
  - 범위: 0.1 ~ 0.3 권장

### 방법 2-A
- `0.5`: 대비 증폭 강도 (Line 88, contrast 방식)
  - 값이 클수록 대비가 더 강해짐
  - 범위: 0.3 ~ 0.7 권장
- `0.3`: 직접 증폭 강도 (Line 95, boost 방식)
  - 값이 클수록 셰이딩이 더 밝아짐
  - 범위: 0.2 ~ 0.5 권장

### 방법 3-A
- `alpha_min`, `alpha_max`: 셰이딩 비중 범위
  - `alpha_min`: 평탄한 영역 셰이딩 비중 (0.2 ~ 0.4 권장)
  - `alpha_max`: 주름 영역 셰이딩 비중 (0.7 ~ 0.9 권장)

## 문제 해결

### Import 오류
- `detail_rendering_methods.py` 파일이 같은 디렉토리에 있는지 확인
- 상대 경로 import가 필요하면 `from .detail_rendering_methods import ...` 사용

### 결과가 부자연스러움
- 하이퍼파라미터 값을 줄여서 시도
- 다른 방법으로 변경해보기

### 주름이 여전히 사라짐
- 여러 방법을 조합하여 사용
- 하이퍼파라미터 값을 증가시켜 시도

## 참고사항

- 모든 방법은 기존 코드를 수정하지 않고 사용할 수 있습니다
- 각 방법은 독립적으로 사용하거나 조합하여 사용할 수 있습니다
- 학습 시와 추론 시 모두 동일하게 사용할 수 있습니다

