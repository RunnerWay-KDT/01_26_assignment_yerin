# U-Net

U-Net은 **Biomedical 분야의 이미지 분할(Image Segmentation)**을 목적으로 제안된 **End-to-End 방식의 Fully Convolutional Network(FCN) 기반 모델**이다. 입력 이미지로부터 픽셀 단위의 Segmentation map을 직접 예측하며, 의료 영상과 같이 **정밀한 경계(Localization)** 가 중요한 문제를 해결하기 위해 설계되었다.

네트워크의 전체 구조가 좌우 대칭의 ‘U’자 형태를 띠고 있어 **U-Net**이라는 이름이 붙여졌다.

![U-Net Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20220614121231/Group14.jpg)  
![U-Net Encoder-Decoder](https://www.researchgate.net/publication/366612721/figure/fig4/AS%3A11431281109636269%401672146713147/Basic-UNet-architecture-showing-encoder-decoder-bridge-network-and-skip-connection.png)

---

## U-Net Architecture 개요

U-Net은 이미지의 **전반적인 컨텍스트(Context) 정보를 얻기 위한 네트워크**와, **정확한 지역화(Localization)를 위한 네트워크**가 대칭적으로 결합된 구조를 가진다. 이는 단순히 “특징을 잘 뽑는 것”이 아니라, **어디에 무엇이 있는지를 정확히 맞히는 것**을 목표로 한다.

전체 구조는 크게 두 부분으로 나뉜다.

- **Contracting Path (수축 단계)**  
  입력 이미지로부터 전역적인 의미 정보를 추출
- **Expanding Path (팽창 단계)**  
  픽셀 단위의 세밀한 Segmentation 결과를 생성

두 경로는 동일한 해상도 단계에서 **Skip Connection**으로 연결되어 있으며, 이는 U-Net 성능의 핵심 요소이다.

---

## Contracting Path — 수축 단계 (Encoder)

![Contracting Path](https://www.researchgate.net/publication/381329058/figure/fig1/AS%3A11431281255932806%401719442418956/Illustration-of-the-U-Net-architecture-The-displayed-U-Net-is-an-encoder-decoder-network.tif)

Contracting Path는 입력 이미지로부터 점점 더 추상적인 특징을 추출하는 역할을 수행하며, 일반적인 CNN 기반 분류 모델과 유사한 구조를 가진다. 다만, 목적은 분류가 아니라 **의미적 특징의 압축(Context Encoding)**이다.

구성은 다음과 같다.

- 3×3 Convolution 연산을 두 차례 반복 (Padding 없음)
- 활성화 함수로 ReLU 사용
- 2×2 Max Pooling (stride = 2)을 통한 Down-sampling
- Down-sampling이 진행될 때마다 채널 수를 2배로 증가

이 과정을 반복하면서 이미지의 공간 해상도는 감소하지만, 더 넓은 영역을 포괄하는 **전역 의미 정보(Global Context)**를 포함하는 특징 표현을 학습하게 된다.

---

## Expanding Path — 팽창 단계 (Decoder)

![Expanding Path](https://miro.medium.com/v2/resize%3Afit%3A1200/1%2AzYrwp34DslR_9wLHMVAITg.png)

Expanding Path는 Contracting Path에서 얻은 저해상도 특징 맵(Coarse Map)으로부터 고해상도의 Segmentation 결과(Dense Prediction)를 복원하는 역할을 한다. 다시 말해, **거친 의미 정보를 픽셀 단위 예측으로 되돌리는 과정**이다.

구성 요소는 다음과 같다.

- 2×2 Up-Convolution(Transposed Convolution)을 통해 Up-sampling 수행
- Up-sampling 시 채널 수는 절반으로 감소
- 3×3 Convolution 연산을 두 차례 반복하며 ReLU 활성화 함수 사용
- 동일한 해상도의 Contracting Path 특징 맵과 Concatenation 수행
- 마지막 레이어에서 1×1 Convolution을 적용하여 픽셀 단위 클래스 예측

이를 통해 의미 정보는 유지하면서, 정확한 위치 정보를 복원할 수 있다.

---

## Skip Architecture (Skip Connection)

![Skip Connection](https://www.mdpi.com/processes/processes-13-01976/article_deploy/html/images/processes-13-01976-g001.png)

U-Net은 FCN에서 제안된 **Skip Architecture** 개념을 확장하여 사용한다. Encoder의 얕은 층에서 추출된 **fine-grained location 정보**와, 깊은 층에서 추출된 **global semantic 정보**를 Decoder에서 결합(concatenation)한다.

이러한 구조를 통해 Segmentation 문제에서 발생하는  
**Localization 정확도와 Context 정보 간의 트레이드오프**를 효과적으로 해결할 수 있다.  
특히 세포, 신경, 병변과 같이 작은 구조물의 경계를 정밀하게 분할하는 데 강점을 가진다.

---

## Architecture Detail

U-Net은 Fully Connected Layer를 사용하지 않는 **Fully Convolutional Network** 구조이며, 전체적으로 약 **23개의 Convolution Layer**로 구성되어 있다.

특징적인 점은 Convolution 연산에서 Padding을 사용하지 않기 때문에,  
**최종 출력 Segmentation map의 크기가 입력 이미지보다 작다**는 점이다.  
이 설계는 이후 설명하는 Overlap-Tile 전략과 직접적으로 연결된다.

---

## Overlap-Tile Input Strategy

![Overlap Tile](https://www.researchgate.net/publication/350484428/figure/fig2/AS%3A1006897591238656%401617074501235/U-Net-Overlap-tile-strategy-for-seamless-segmentation-of-arbitrary-large-images-here.ppm)

U-Net은 FCN 구조이기 때문에 입력 이미지 크기에 제약이 없지만, 고해상도 의료 영상을 그대로 입력할 경우 메모리 사용량이 급격히 증가한다. 이를 해결하기 위해 **Overlap-Tile 전략**이 제안되었다.

- 입력 이미지를 여러 개의 타일(Tile, Patch)로 분할
- 하나의 입력 타일로부터 중앙 영역의 Segmentation 결과를 예측
- 다음 타일은 이전 타일과 일부 영역이 겹치도록 설정

이를 통해 경계 영역에서도 안정적인 예측이 가능하며, 대형 의료 이미지에 대해서도 효율적인 추론이 가능하다.

---

## Boundary Handling — Mirror Extrapolation

![Mirror Extrapolation](https://miro.medium.com/v2/resize%3Afit%3A700/0%2AhowsOWgKADN54VBl.png)

이미지 경계 부분에서 발생하는 Segmentation 성능 저하를 방지하기 위해, U-Net은 Zero Padding 대신 **Mirror Extrapolation** 기법을 사용한다. 이미지의 경계 부분을 거울처럼 반사하여 확장함으로써, 경계 픽셀에서도 자연스러운 예측이 가능하도록 한다.

---

## Touching Cells Separation

![Touching Cells](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41592-018-0261-2/MediaObjects/41592_2018_261_Fig1_HTML.png)

Biomedical Segmentation에서 중요한 과제 중 하나는 **동일 클래스 객체들이 서로 접촉한 경우를 분리하는 것**이다. U-Net은 이를 해결하기 위해 **Weight Map 기반 Loss**를 도입하였다.

- 세포 간 경계 픽셀에 더 큰 가중치 부여
- 각 픽셀에 대해
  - 가장 가까운 세포 경계까지의 거리
  - 두 번째로 가까운 세포 경계까지의 거리  
  를 기반으로 Weight Map 생성

이를 통해 모델이 경계 영역 학습에 더 집중하도록 유도한다.

---

## Summary

U-Net은 FCN 구조를 기반으로 Up-sampling과 Skip Architecture를 결합한 이미지 분할 모델이다. 적은 양의 학습 데이터에서도 강력한 Data Augmentation과 구조적 설계를 통해 다양한 Biomedical Image Segmentation 문제에서 우수한 성능을 보였다. 특히 Localization과 Context 정보를 동시에 보존할 수 있다는 점에서 의료 영상 분할 분야의 대표적인 베이스라인 모델로 활용되고 있다.

## Dataset

본 실험에서는 JSRT Lung Segmentation 데이터셋을 사용하였습니다. 데이터 구조는 다음과 같습니다.

datasets/jsrt/content/jsrt/
- cxr/   : Chest X-ray images (PNG)
- masks/ : Lung segmentation masks (PNG)

각 이미지와 마스크는 파일명 기준으로 1:1 대응됩니다.