# Few-shot Task-incremental Learning: Methods, Challenges, and Future Directions
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)[![Few-shot Task-incremental Learning: Methods, Challenges, and Future Directions](https://awesome.re/badge.svg)](https://awesome.re)
## Table of Contents
- [Few-shot Task-incremental Learning: Methods, Challenges, and Future Directions](#few-shot-task-incremental-learning-methods-challenges-and-future-directions)
  - [Table of Contents](#table-of-contents)
  - [0. Overview](#0-overview)
  - [1. Data-Based Approaches](#1-data-based-approaches)
    - [1.1 Replay-Based Approaches](#11-replay-based-approaches)
      - [1.1.1 Brain-inspired Replay](#111-brain-inspired-replay)
          - [Brain-inspired Replay 2025](#brain-inspired-replay-2025)
          - [Brain-inspired Replay 2024](#brain-inspired-replay-2024)
          - [Brain-inspired Replay 2023](#brain-inspired-replay-2023)
          - [Brain-inspired Replay 2022](#brain-inspired-replay-2022)
          - [Brain-inspired Replay 2021](#brain-inspired-replay-2021)
          - [Brain-inspired Replay 2020](#brain-inspired-replay-2020)
      - [1.1.2 Feature Replay](#112-feature-replay)
          - [Pruning During Training CNNs 2024](#pruning-during-training-cnns-2024)
          - [Pruning During Training CNNs 2023](#pruning-during-training-cnns-2023)
          - [Pruning During Training CNNs 2022](#pruning-during-training-cnns-2022)
          - [Pruning During Training CNNs 2021](#pruning-during-training-cnns-2021)
          - [Pruning During Training CNNs 2020](#pruning-during-training-cnns-2020)
          - [Pruning During Training CNNs 2019](#pruning-during-training-cnns-2019)
          - [Pruning During Training CNNs 2018 and earlier](#pruning-during-training-cnns-2018-and-earlier)
      - [1.1.3 Generative Replay](#113-generative-replay)
          - [Pruning Other Models](#pruning-other-models)
          - [Pruning After Training CNNs 2024](#pruning-after-training-cnns-2024)
          - [Pruning After Training CNNs 2023](#pruning-after-training-cnns-2023)
          - [Pruning After Training CNNs 2022](#pruning-after-training-cnns-2022)
          - [Pruning After Training CNNs 2021](#pruning-after-training-cnns-2021)
          - [Pruning After Training CNNs 2020](#pruning-after-training-cnns-2020)
          - [Pruning After Training CNNs 2019](#pruning-after-training-cnns-2019)
          - [Pruning After Training CNNs 2018](#pruning-after-training-cnns-2018)
          - [Pruning After Training CNNs 2017 and earlier](#pruning-after-training-cnns-2017-and-earlier)
      - [1.1.4 Pseudo-scenarios Replay](#114-pseudo-scenarios-replay)
          - [Pruning After Training ViTs 2024](#pruning-after-training-vits-2024)
          - [Pruning After Training ViTs 2023](#pruning-after-training-vits-2023)
          - [Pruning After Training ViTs 2022](#pruning-after-training-vits-2022)
      - [1.1.5 Raw-data Replay](#115-raw-data-replay)
          - [Pruning After Training BERTs 2023](#pruning-after-training-berts-2023)
          - [Pruning After Training BERTs 2022](#pruning-after-training-berts-2022)
          - [Pruning After Training BERTs 2021](#pruning-after-training-berts-2021)
          - [Pruning After Training BERTs 2020](#pruning-after-training-berts-2020)
          - [Pruning After Training BERTs 2019](#pruning-after-training-berts-2019)
    - [1.2 Data-Augmentation-Based Approaches](#12-data-augmentation-based-approaches)
          - [Pruning After Training LLMs 2024](#pruning-after-training-llms-2024)
      - [1.2.1 Data Augmentation](#121-data-augmentation)
          - [Pruning After Training LLMs 2023](#pruning-after-training-llms-2023)
          - [Pruning After Training Diffusion Models 2023](#pruning-after-training-diffusion-models-2023)
          - [Pruning After Training VLMs 2024](#pruning-after-training-vlms-2024)
          - [Pruning After Training VLMs 2023](#pruning-after-training-vlms-2023)
          - [Pruning After Training VLMs 2022](#pruning-after-training-vlms-2022)
      - [1.2.2 Feature Augmentation](#122-feature-augmentation)
          - [Post Training 2024](#post-training-2024)
          - [Post Training 2023](#post-training-2023)
          - [Post Training 2022](#post-training-2022)
          - [Post Training 2021](#post-training-2021)
          - [Post Training 2021](#post-training-2021-1)
          - [Post Training 2021](#post-training-2021-2)
  - [2. model Based Approaches](#2-model-based-approaches)
    - [2.1 Architecture-Based Approches](#21-architecture-based-approches)
      - [2.1.1 Attention-Based](#211-attention-based)
      - [2.1.2 Dynamic network structure-based](#212-dynamic-network-structure-based)
          - [Post Training 2021](#post-training-2021-3)
  - [3. Optimization Based Approaches](#3-optimization-based-approaches)
    - [3.1 Gradient-Based Approaches](#31-gradient-based-approaches)
      - [3.1.1 Function Regularization](#311-function-regularization)
      - [3.1.2 Weight Regularization](#312-weight-regularization)
      - [3.1.3 Gradient Space](#313-gradient-space)
      - [3.1.4 Loss Function](#314-loss-function)
    - [3.2 Tuning-Based Approaches](#32-tuning-based-approaches)
      - [3.2.1 Adapter-Based](#321-adapter-based)
      - [3.2.2 Instruct Tuning](#322-instruct-tuning)
      - [3.2.3 Prompt-Based](#323-prompt-based)
      - [3.2.4 Prefix-Tuning](#324-prefix-tuning)
  - [4. Survey of Pruning](#4-survey-of-pruning)
    - [Survey of Pruning 2024](#survey-of-pruning-2024)
    - [Survey of Pruning 2023](#survey-of-pruning-2023)
    - [Survey of Pruning 2022](#survey-of-pruning-2022)
    - [Survey of Pruning 2021](#survey-of-pruning-2021)
    - [Survey of Pruning 2020](#survey-of-pruning-2020)
    - [Survey of Pruning 2019 and earlier](#survey-of-pruning-2019-and-earlier)
  - [5. Other Works](#5-other-works)
    - [Papers](#papers)

## 0. Overview
The repo includes the ongoing updates of representative few-shot task incremental learning papers and open-source codes.  

**Taxonomy**: In our survey, we provide a comprehensive review of the state-of-the-art in deep neural network pruning, which we categorize along five orthogonal axes: Universal/Specific Speedup, When to Prune, Pruning Criteria, Learn to Prune, and Fusion of Pruning and Other Techniques. 

<p align="center">
  <img src=taxonomy.png width="500">
</p>


## 1. Data-Based Approaches
### 1.1 Replay-Based Approaches
#### 1.1.1 Brain-inspired Replay
###### Brain-inspired Replay 2025
| No. | Title   | Venue | Algorithm Name | Code | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Learn by Reasoning: Analogical Weight Generation for Few-Shot Class-Incremental Learning](https://arxiv.org/pdf/2503.21258?) | arXiv | - | - | 2025 |

###### Brain-inspired Replay 2024
| No. | Title   | Venue | Algorithm Name | Code | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Brain-inspired fast-and slow-update prompt tuning for few-shot class-incremental learning](https://ieeexplore.ieee.org/abstract/document/10682795) | TNNLS | FSPT-FSCIL | [GitHub](https://github.com/qihangran/FSPT-FSCIL) | 2024 |
| 02 | [MgSvF: Multi-Grained Slow versus Fast Framework for Few-Shot Class-Incremental Learning](https://openreview.net/forum?id=uVcDssQff) | TPAMI | SvF | - | 2024 |
| 03 | [SHARP: Sparsity and Hidden Activation RePlay for Neuro-Inspired Continual Learning](https://ieeexplore.ieee.org/abstract/document/10644996) | ICDL | SHARP | [GitHub](https://github.com/BurakGurbuz97/SHARP-Continual-Learning) | 2024 |


###### Brain-inspired Replay 2023
| No. | Title   | Venue | Algorithm Name | Code | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Few-shot class-incremental learning via class-aware bilateral distillation](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Few-Shot_Class-Incremental_Learning_via_Class-Aware_Bilateral_Distillation_CVPR_2023_paper.pdf) | CVPR | CABD | [GitHub](https://github.com/LinglanZhao/BiDistFSCIL) | 2023 |
| 02 | [Class-incremental learning using generative experience replay based on time-aware regularization](https://arxiv.org/pdf/2310.03898) | arXiv | - | - | 2023 |


###### Brain-inspired Replay 2022
| No. | Title   | Venue | Algorithm Name | Code | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Semantics-driven generative replay for few-shot class incremental learning](https://dl.acm.org/doi/abs/10.1145/3503161.3548160)| Proc ACM Int Conf Multimed | FSIL-GAN | - | Image Classification | 2022 |
| 02 | [Memory-based label-text tuning for few-shot class-incremental learning](https://openreview.net/forum?id=fOsN52jn25l) | arXiv |  M-FSCIL | - | 2022 |


###### Brain-inspired Replay 2021
| No. | Title   | Venue | Algorithm Name | Code | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Triple-memory networks: A brain-inspired method for continual learning](https://ieeexplore.ieee.org/abstract/document/9540230/) | TNNLS | TMNs | - |  2021 |
| 02 | [Few-shot continual learning: A brain-inspired approach](https://arxiv.org/pdf/2104.09034) | arXiv | TSC | - | 2021 |


###### Brain-inspired Replay 2020
| No. | Title   | Venue | Algorithm Name | Code | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Remind your neural network to prevent catastrophic forgetting](https://link.springer.com/chapter/10.1007/978-3-030-58598-3_28) | ECCV | REMIND | [GitHub](https://github.com/tyler-hayes/REMIND) |  2020 |
| 02 | [Brain-inspired replay for continual learning with artificial neural networks](https://www.nature.com/articles/s41467-020-17866-2) | Nat Commun | BIR | - | 2020 |



#### 1.1.2 Feature Replay
###### Feature Replay 2024
| No. | Title | Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Learning prompt with distribution-based feature replay for few-shot class-incremental learning](https://arxiv.org/pdf/2401.01598) | arXiv | LP-DiF | [GitHub](https://github.com/1170300714/LP-DiF) | 2024 |
| 02 | [Few-Shot Class-Incremental Learning via Cross-Modal Alignment with Feature Replay](https://link.springer.com/chapter/10.1007/978-981-97-8487-5_2) | PRCV | - | - | 2024 |
| 03 | [SHARP: Sparsity and Hidden Activation RePlay for Neuro-Inspired Continual Learning](https://ieeexplore.ieee.org/abstract/document/10644996) | ICDL | SHARP | [GitHub](https://github.com/BurakGurbuz97/SHARP-Continual-Learning) | 2024 |
| 02 | [Prototype-guided memory replay for continual learning](https://ieeexplore.ieee.org/abstract/document/10058177) | TNNLS‌ | PMR | - | 2024 |
###### Feature Replay 2023
| No. | Title | Venue | Algorithm Name | Code |Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Prototype reminiscence and augmented asymmetric knowledge aggregation for non-exemplar class-incremental learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Prototype_Reminiscence_and_Augmented_Asymmetric_Knowledge_Aggregation_for_Non-Exemplar_Class-Incremental_ICCV_2023_paper.pdf) | ICCV | NECIL | [GitHub](https://shiwuxuan.github.io/PRAKA-project) | 2023 |
| 02 | [Saving 100x storage: Prototype replay for reconstructing training sample distribution in class-incremental semantic segmentation](https://proceedings.neurips.cc/paper_files/paper/2023/file/708e0d691a22212e1e373dc8779cbe53-Paper-Conference.pdf) | NeurIPS | STAR | [GitHub](https://github.com/jinpeng0528/STAR) | 2023 |
| 03 | [Few shot class incremental learning via efficient prototype replay and calibration](https://www.mdpi.com/1099-4300/25/5/776) | ENTROPY-SWITZ‌ | EPRC  | - | 2023 |

###### Feature Replay 2022
| No. | Title | Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Continual learning with foundation models: An empirical study of latent replay](https://proceedings.mlr.press/v199/ostapenko22a/ostapenko22a.pdf) | CoLLAs‌| latent ER | [GitHub](https://github.com/oleksost/latent_CL) | 2022 |
| 02 | [Semantics-driven generative replay for few-shot class incremental learning](https://dl.acm.org/doi/abs/10.1145/3503161.3548160) | ACM MM‌ | GAN | - | 2022 |
| 03 | [Prompt-based prototypical framework for continual relation extraction](https://ieeexplore.ieee.org/abstract/document/9860068) | TASLP‌ | CRE |- | 2022 |

###### Feature Replay 2021
| No. | Title | Venue  | Algorithm Name | Code  | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Generative feature replay for class-incremental learning](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Liu_Generative_Feature_Replay_for_Class-Incremental_Learning_CVPRW_2020_paper.pdf) | CVPR | CCA |[GitHub](https://github.com/xialeiliu/GFR-IL) | 2021 |


#### 1.1.3 Generative Replay
###### Generative Replay 2025
| No. | Title   | Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Continual Learning of Personalized Generative Face Models with Experience Replay](https://ieeexplore.ieee.org/abstract/document/10944168) | WACV | - | [GitHub](https://anniedde.github.io/personalizedcontinuallearning.github.io/) | 2025 |
| 02 | [AnchorInv: Few-Shot Class-Incremental Learning of Physiological Signals via Feature Space-Guided Inversion](https://ojs.aaai.org/index.php/AAAI/article/view/33563/35718) | AAAI | - | [GitHub](https://github.com/chenqi-li/anchorinv) | 2025 |


###### Generative Replay 2024
| No. | Title   | Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Continual offline reinforcement learning via diffusion-based dual generative replay](https://arxiv.org/pdf/2404.10662) | arXiv | CuGRO | [GitHub](https://github.com/NJU-RL/CuGRO) | 2024 |
| 02 | [Clip with generative latent replay: a strong baseline for incremental learning](https://arxiv.org/pdf/2407.15793?) | arXiv | CGIL | [GitHub](https://github.com/aimagelab/mammoth) | 2024 |
| 03 | [Few-shot task learning through inverse generative modeling](https://proceedings.neurips.cc/paper_files/paper/2024/file/) | NeurIPS | - | - | 2024 |
| 04 | [General federated class-incremental learning with lightweight generative replay](https://ieeexplore.ieee.org/abstract/document/10612802/) | IEEE IoT-J | GenFCIL | - | 2024 |

###### Generative Replay 2023
| No. | Title | Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Task-aware information routing from common representation space in lifelong learning](https://arxiv.org/pdf/2302.11346) | arXiv | TAMiL | [GitHub](https://github.com/NeurAI-Lab/TAMiL) | 2023 |
| 02 | [Class-incremental learning using diffusion model for distillation and replay](https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/Jodelet_Class-Incremental_Learning_Using_Diffusion_Model_for_Distillation_and_Replay_ICCVW_2023_paper.pdf) | ICCV | SDDR | - | 2023 |


###### Generative Replay 2022
| No. | Title | Type | Algorithm Name | Code |Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Few-shot class-incremental learning via entropy-regularized data-free replay](https://link.springer.com/chapter/10.1007/978-3-031-20053-3_9) | ECCV | - | - | 2022 |
| 02 | [Memory replay with data compression for continual learning](https://arxiv.org/pdf/2202.06592) | arXiv | MRDC | - | 2022 |
| 03 | [Semantics-driven generative replay for few-shot class incremental learning](https://dl.acm.org/doi/abs/10.1145/3503161.3548160) | Proc ACM Int Conf Multimed | - | - | 2022 |

###### Generative Replay 2021
| No. | Title | Type | Algorithm Name | Code| Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Triple-memory networks: A brain-inspired method for continual learning](https://ieeexplore.ieee.org/abstract/document/9540230/) | TNNLS | TMNs | - | 2021 |
 
 
#### 1.1.3 Pseudo-scenarios Reply
###### Pseudo-scenarios Reply 2025
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Pseudo Informative Episode Construction for Few-Shot Class-Incremental Learning](https://ojs.aaai.org/index.php/AAAI/article/view/33729)  | AAAI  | PIEC  | - | 2025 |
| 02 | [SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection](https://openaccess.thecvf.com/content/CVPR2025/papers/Tran_SimLTD_Simple_Supervised_and_Semi-Supervised_Long-Tailed_Object_Detection_CVPR_2025_paper.pdf)  | CVPR   | SimLTD   | - | 2025 |

###### Pseudo-scenarios Reply 2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [PASS-Net: A Pseudo Classes and Stochastic Classifiers Based Network for Few-Shot Class-Incremental Automatic Modulation Classification](https://ieeexplore.ieee.org/abstract/document/10684455/)  | TWC   | PASS-Net  | - | 2024 |
| 02 | [M2SD: Multiple Mixing Self-Distillation for Few-shot Class-Incremental Learning](https://ojs.aaai.org/index.php/AAAI/article/view/28129)  | AAAI    | M2SD    | - | 2024 |

###### Pseudo-scenarios Reply 2023
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Serial Contrastive Knowledge Distillation for Continual Few-shot Relation Extraction](https://arxiv.org/pdf/2305.06616)  | arXiv   | SCKD  | [GitHub](https://github.com/nju-websoft/SCKD) | 2023 |
| 02 | [Evolving Dictionary Representation for Few-shot Class-incremental Learning](https://arxiv.org/pdf/2305.01885)  | arXiv    | D-FSCIL    | - | 2023 |
| 03 | [Learning with Fantasy: Semantic-Aware Virtual Contrastive Constraint for Few-Shot Class-Incremental Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_Learning_With_Fantasy_Semantic-Aware_Virtual_Contrastive_Constraint_for_Few-Shot_Class-Incremental_CVPR_2023_paper.pdf)  | CVPR    | SAVC    | [GitHub](https://github.com/zysong0113/SAVC)  | 2023 |

###### Pseudo-scenarios Reply 2022
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 |  [Forward compatible few-shot class-incremental learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Forward_Compatible_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf)  | CVPR   | BGM  | - | 2022 |
| 02 | [Forward compatible few-shot class-incremental learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Forward_Compatible_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf)  | CVPR    | FACT    | [GitHub](https://github.com/zhoudw-zdw/CVPR22-Fact) | 2022 |
| 03 | [Few-shot class-incremental learning by sampling multi-phase tasks](https://ieeexplore.ieee.org/abstract/document/9864267) | TPAMI    | LIMIT    | - | 2022 |


###### Pseudo-scenarios Reply 2021 and earlier
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 |  [Few-shot incremental learning with continually evolved classifiers](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Few-Shot_Incremental_Learning_With_Continually_Evolved_Classifiers_CVPR_2021_paper.pdf)  | CVPR   | CEC  | - | 2021 |
| 02 | [LFPT5: A unified framework for lifelong few-shot language learning based on prompt tuning of T5](https://arxiv.org/pdf/2110.07298)  | arXiv    | LFPT5 | [GitHub](https://github.com/qcwthu/Lifelong-Fewshot-Language-Learning) | 2021 |
| 03 | [Self-supervised label augmentation via input transformations](https://proceedings.mlr.press/v119/lee20c/lee20c.pdf) | ICML    | SLA  | [GitHub](https://github.com/hankook/SLA) | 2020 |


#### 1.1.4 Raw-data replay
###### Raw-data replay 2025
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Continual Learning of Personalized Generative Face Models with Experience Replay](https://ieeexplore.ieee.org/abstract/document/10944168) | WACV | - | [GitHub](https://anniedde.github.io/personalizedcontinuallearning.github.io/) | 2025 |


###### Raw-data replay 2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [InsCL: A data-efficient continual learning paradigm for fine-tuning large language models with instructions](https://arxiv.org/abs/2403.11435) | arXiv | InsCL | - | 2024 |
| 02 | [Learning to learn for few-shot continual active learning](https://link.springer.com/content/pdf/10.1007/s10462-024-10924-x.pdf) | Artificial Intelligence Review | Meta-CAL | [GitHub](https://pytorch.org/1.10.0+cu113) | 2024 |

###### Raw-data replay 2023
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Task-aware information routing from common representation space in lifelong learning](https://arxiv.org/pdf/2302.11346) | arXiv | TAMiL | [GitHub](https://github.com/NeurAI-Lab/TAMiL) | 2023 |

###### Raw-data replay 2022
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Memory replay with data compression for continual learning](https://arxiv.org/pdf/2202.06592) | arXiv | MRDC | - | 2022 |
| 02 | [Incremental meta-learning via episodic replay distillation for few-shot image recognition](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Wang_Incremental_Meta-Learning_via_Episodic_Replay_Distillation_for_Few-Shot_Image_Recognition_CVPRW_2022_paper.pdf) | CVPR | ERD | - | 2022 |
| 03 | [Fine-tuned language models are continual learners](https://arxiv.org/pdf/2205.12393) | arXiv | CT0 | [GitHub](https://github.com/ThomasScialom/T0_continual_learning) | 2022 |


###### Raw-data replay 2021
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Generalized and incremental few-shot learning by explicit learning and calibration without forgetting](https://openaccess.thecvf.com/content/ICCV2021/papers/Kukleva_Generalized_and_Incremental_Few-Shot_Learning_by_Explicit_Learning_and_Calibration_ICCV_2021_paper.pdf) | ICCV | LCwoF | [GitHub](https://github.com/annusha/LCwoF) | 2021 |
| 02 |  [Few-shot incremental learning with continually evolved classifiers](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Few-Shot_Incremental_Learning_With_Continually_Evolved_Classifiers_CVPR_2021_paper.pdf)  | CVPR   | CEC  | - | 2021 |

### 1.2 Data-Augmentation-Based Approaches 
#### 1.2.1 Data Augmentation
###### Data Augmentation 2025
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Generalized and incremental few-shot learning by explicit learning and calibration without forgetting](https://openaccess.thecvf.com/content/ICCV2021/papers/Kukleva_Generalized_and_Incremental_Few-Shot_Learning_by_Explicit_Learning_and_Calibration_ICCV_2021_paper.pdf) | ICCV | LCwoF | [GitHub](https://github.com/annusha/LCwoF) | 2021 |
| 02 |  [Few-shot incremental learning with continually evolved classifiers](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Few-Shot_Incremental_Learning_With_Continually_Evolved_Classifiers_CVPR_2021_paper.pdf)  | CVPR   | CEC  | - | 2021 |

###### Pruning After Training CNNs 2017 and earlier
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Optimal Brain Damage](https://proceedings.neurips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf) | NIPS | `W` | OBD | - | Image Classification | 1989 |
| 02 | [Second Order Derivatives for Network Pruning: Optimal Brain Surgeon](https://proceedings.neurips.cc/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf) | NIPS | `W` | OBS | - | Image Classification | 1992 |
| 03 | [Structured Pruning of Deep Convolutional Neural Networks](https://arxiv.org/pdf/1512.08571) | arXiv | `C` | - | - | Image Classification | 2015 |
| 04 | [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) | ICLR **(Best)** | `W`  | - |[Caffe(Author)](https://github.com/songhan/Deep-Compression-AlexNet) | Image Classification | 2016 |
| 05 | [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342) | ICCV&TPAMI | `F` | ThiNet | [Caffe(Author)](https://github.com/Roll920/ThiNet), [PyTorch(3rd)](https://github.com/tranorrepository/reprod-thinet) | Image Classification | 2017&2019 |
| 06 | [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440) | ICLR | `F` | - | [PyTorch](https://github.com/jacobgil/pytorch-pruning/tree/master) | Image Classification | 2017 |
| 07 | [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) | ICLR    | `F`  | PFEC | [PyTorch(3rd)](https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/imagenet/l1-norm-pruning) | Image Classification | 2017 |
| 08 | [Channel pruning for accelerating very deep neural networks](https://arxiv.org/abs/1707.06168) | ICCV | `C` | - | [Caffe(Author)](https://github.com/yihui-he/channel-pruning) |Image Classification&Object Detection | 2017 |



#### 1.1.4 Pseudo-scenarios Replay
###### Pruning After Training ViTs 2024
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Fast and Controllable Post-training Sparsity: Learning Optimal Sparsity Allocation with Global Constraint in Minutes](https://arxiv.org/abs/2203.04570) | AAAI | `W` | FCPTS | - | Image Classification&Object Detection | 2024 |
| 02 | [UPDP: A Unified Progressive Depth Pruner for CNN and Vision Transformer](https://arxiv.org/pdf/2401.06426v1#page=3.05) | AAAI | `L` | UPDP | - | Image Classification&Object Detection | 2024 |
| 03 | [Pruning Self-attentions into Convolutional Layers in Single Path](https://arxiv.org/abs/2111.11802) | TPAMI | `H` | SPViT |  [PyTorch](https://github.com/ziplab/SPViT) | Image Classification&Object Detection | 2024 |

###### Pruning After Training ViTs 2023
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [X-Pruner: eXplainable Pruning for Vision Transformers](https://arxiv.org/abs/2303.04935) | CVPR | `CH` | X-Pruner | [Pytorch(Author)](https://github.com/vickyyu90/XPruner) | Image Classification | 2023 |
| 02 | [Global Vision Transformer Pruning with Hessian-Aware Saliency](https://arxiv.org/abs/2110.04869) | CVPR | `CH` | NViT | - | Image Classification | 2023 |
| 03 | [Pruning Parameterization with Bi-level Optimization for Efficient Semantic Segmentation on the Edge](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Pruning_Parameterization_With_Bi-Level_Optimization_for_Efficient_Semantic_Segmentation_on_CVPR_2023_paper.pdf) | CVPR | `W` | STE | - | semantic Segmentation | 2023 |
| 04 | [Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models](https://arxiv.org/abs/2306.10460) | ICML | `W` | ISP | [Pytorch(Author)](https://github.com/VITA-Group/instant_soup) | Image Classification&NLP | 2023 |

###### Pruning After Training ViTs 2022
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Width & Depth Pruning for Vision Transformers](https://cdn.aaai.org/ojs/20222/20222-13-24235-1-2-20220628.pdf) | AAAI | `C` | WDPruning | [Pytorch(Author)](https://github.com/andyrull/width-and-Depth-pruning-for-Vision-Transformer) | Image Classification | 2022 |
| 02 | [SAViT: Structure-Aware Vision Transformer Pruning via Collaborative Optimization](https://cdn.aaai.org/ojs/20222/20222-13-24235-1-2-20220628.pdf) | NeurIPS | `CHE` | SAViT | [Pytorch(Author)](https://github.com/hikvision-research/SAViT) | Image Classification&object detection | 2022 |
| 03 | [VTC-LFC: Vision Transformer Compression with Low-Frequency Components](https://papers.neurips.cc/paper_files/paper/2022/file/5a8177df23bdcc15a02a6739f5b9dd4a-Paper-Conference.pdf) | NeurIPS | `C` | VTC-LFC | [Pytorch(Author)](https://github.com/Daner-Wang/VTC-LFC) | Image Classification | 2022 |
| 04 | [CP-ViT: Cascade Vision Transformer Pruning via Progressive Sparsity Prediction](https://arxiv.org/abs/2203.04570) | arXiv | `H` | CP-ViT  | - | Image Classification | 2022 |
| 05 | [Unified Visual Transformer Compression](https://arxiv.org/abs/2203.08243) | ICLR | `H` | UVC  | [Pytorch(Author)](https://github.com/VITA-Group/UVC) | Image Classification | 2022 |

#### 1.1.5 Raw-data Replay
###### Pruning After Training BERTs 2023
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation](https://proceedings.mlr.press/v202/li23ap/li23ap.pdf) | ICML |  `H` | LoSparse | [PyTorch(Author)](https://github.com/yxli2123/LoSparse) | NLP | 2023|
| 02 | [Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models](https://arxiv.org/abs/2306.10460) | ICML | `W` | ISP | [Pytorch(Author)](https://github.com/VITA-Group/instant_soup) | Image Classification&NLP | 2023 |
| 03 | [Gradient-Free Structured Pruning with Unlabeled Data](https://arxiv.org/pdf/2204.00408.pdf) | ICML |  `F` | KCM | - | NLP | 2023|
| 04 | [The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter](https://arxiv.org/abs/2306.03805) | arXiv |  `W`&N:M | - | [Pytorch(Author)](https://github.com/VITA-Group/essential_sparsity?tab=readme-ov-file) | NLP | 2023|

###### Pruning After Training BERTs 2022
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Structured Pruning Learns Compact and Accurate Models](https://arxiv.org/pdf/2204.00408.pdf) | ACL |  `LH` | CoFi | [PyTorch(Author)](https://github.com/princeton-nlp/CoFiPruning)  | Natural Language Understanding | 2022|
| 02 | [From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language Model Compression](https://arxiv.org/abs/2112.07198) | AAAI |  `WH` | CAP | [PyTorch(Author)](https://github.com/alibaba/AliceMind/tree/main/ContrastivePruning)  | NLP | 2022|
| 03 | [PLATON: Pruning Large Transformer Models with Upper Confidence Bound of Weight Importance](https://arxiv.org/abs/2206.12562) | ICML |  `WC` | PLATON | [PyTorch(Author)](https://github.com/QingruZhang/PLATON)  | Natural Language Understanding&Question Answering&Image Classification | 2022|
| 04 | [Parameter-Efficient Sparsity for Large Language Models Fine-Tuning](https://arxiv.org/pdf/2205.11005.pdf) | IJCAI | `W` | PST |  [PyTorch(Author)](https://github.com/yuchaoli/pst) | Language Modeling | 2022|
| 05 | [The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models](https://arxiv.org/pdf/2203.07259.pdf) | EMNLP |  `W` | oBERT | [PyTorch(Author)](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT)| Natural Language Understanding | 2022|
| 06 | [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://arxiv.org/abs/2208.11580) | NeurIPS | `W` | ExactOBS  | [PyTorch(Author)](https://github.com/IST-DASLab/OBC) | Image Classification&Object Detection&Question Answering | 2022 |

###### Pruning After Training BERTs 2021
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Super Tickets in Pre-Trained Language Models: From Model Compression to Improving Generalization](https://arxiv.org/abs/2105.12002) | ACL | `W` | super tickets | [PyTorch(Author)](https://github.com/cliang1453/super-structured-lottery-tickets) | Language Understanding | 2021 | 
| 02 | [Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks](https://arxiv.org/abs/2102.08124)  | NeurIPS | N:M | AdaPrune | [PyTorch(Author)](https://github.com/papers-submission/structured_transposable_masks) | Image Classification | 2021 |
| 03 | [Prune Once for All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754)  | NeurIPS Workshop | `W` | OFA | [PyTorch(Author)](https://github.com/IntelLabs/Model-Compression-Research-Package) | NLP | 2021 | 
| 04 | [BERT Busters: Outlier Dimensions that Disrupt Transformers](https://arxiv.org/abs/2105.06990) | ACL | `W` | - | - | NLP | 2021 | 
| 05 | [PARP: Prune, Adjust and Re-Prune for Self-Supervised Speech Recognition](https://arxiv.org/abs/2106.05933) | NeurIPS | `W` | PARP | -| Speach Recognition | 2021 | 
| 06 | [Parameter-Efficient Transfer Learning with Diff Pruning](https://arxiv.org/abs/2012.07463) | ACL | `M` | Diff Pruning | [PyTorch(Author)](https://github.com/dguo98/DiffPruning) | NLP | 2021 | 
| 07 | [EarlyBERT: Efficient BERT training via early-bird lottery tickets](https://arxiv.org/abs/2101.00063) | ACL-IJCNLP| `H` | EarlyBERT | [PyTorch(Author)](https://github.com/VITA-Group/EarlyBERT) | NLP | 2021 |
| 08 | [The Lottery Ticket Hypothesis for Pre-trained BERT Networks](https://arxiv.org/abs/2007.12223) | ICML | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/BERT-Tickets) | Language Modeling | 2021 |
| 09 | [Structured Pruning of Large Language Models](https://arxiv.org/abs/1910.04732) | arXiv | `W` | FLOP | [PyTorch(Author)](https://github.com/asappresearch/flop) | NLP classification | 2021 | 
| 10 | [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378) | arXiv | `W` | - | - | Image Classification&Image Segmentation and Detection&Language Modeling&Language Translation | 2021 |
| 11 | [Differentiable Subset Pruning of Transformer Heads](https://arxiv.org/abs/2108.04657) | TACL | `H` | - | [PyTorch(Author)](https://github.com/rycolab/differentiable-subset-pruning) | NLP | 2021 |

###### Pruning After Training BERTs 2020
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 03 | [Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers](https://arxiv.org/abs/2002.11794)| ICML | `W`| - | - | NLP | 2020 |
| 04 | [When BERT Plays the Lottery, All Tickets Are Winning](https://arxiv.org/abs/2005.00561) | EMNLP | `W` | - | [PyTorch(Author)](https://github.com/sai-prasanna/bert-experiments) | Language Modeling | 2020 |
| 05 | [LadaBERT: Lightweight Adaptation of BERT through Hybrid Model Compression](https://arxiv.org/abs/2004.04124) | COLING | `W` | - | - | NLP(Sentiment Classification,Natural Language Inference,Pairwise Semantic Equivalence) | 2020 |
| 06 | [Pruning Redundant Mappings in Transformer Models via Spectral-Normalized Identity Prior](https://arxiv.org/abs/2010.01791) | EMNLP| `H` | - | - | NLP | 2020 |
| 07 | [Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning](https://arxiv.org/abs/2002.08307) | Rep4NLP| `W` | - | - | NLP | 2020 |

###### Pruning After Training BERTs 2019
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Reweighted Proximal Pruning for Large-Scale Language Representation](http://arxiv.org/abs/1909.12486) | arXiv| `Other`  | -  | - | NLP | 2019 |
| 02 | [Efficient Transformer-based Large Scale Language Representations using Hardware-friendly Block Structured Pruning](https://arxiv.org/abs/2009.08065) | EMNLP| `Other`| - | - | NLP | 2019 |

### 1.2 Data-Augmentation-Based Approaches
###### Pruning After Training LLMs 2024
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2305.18403) | ACL | `CH` | LoRAPrune | [PyTorch(Author)](https://github.com/aim-uofa/LoRAPrune)  |Language Modeling&Classification | 2024|
| 02 | [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695) | ICLR | `W` |  Wanda | [PyTorch(Author)](https://github.com/locuslab/wanda)  | Language Modeling&Classification | 2024|
| 03 | [SliceGPT: Compress Large Language Models by Deleting Rows and Columns](https://arxiv.org/abs/2401.15024) | ICLR | `CH` | SliceGPT | [PyTorch(Author)](https://github.com/microsoft/TransformerCompression)  | Language Modeling&Classification | 2024|
| 04 | [Fluctuation-based Adaptive Structured Pruning for Large Language Models](https://arxiv.org/abs/2312.11983) | AAAI | `CH` | FLAP | [PyTorch(Author)](https://github.com/CASIA-IVA-Lab/FLAP)  | Language Modeling&Classification | 2024|
| 05 | [BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation](https://arxiv.org/abs/2402.16880) | arXiv | `B` | BESA | [PyTorch(Author)](https://github.com/OpenGVLab/LLMPrune-BESA) |Language Modeling&Classification | 2024|
| 06 | [APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference](https://arxiv.org/abs/2401.12200) | ICML | `HC` | APT | [PyTorch(Author)](https://github.com/roim1998/apt) |Classification | 2024|
| 07 | [Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning](https://arxiv.org/abs/2310.06694) | ICLR | `CH` | Sheared LLaMA | [PyTorch(Author)](https://github.com/princeton-nlp/LLM-Shearing)  | Language Modeling&Classification | 2024|
| 08 | [Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes](https://arxiv.org/abs/2402.05406) | arXiv | `CH` | Bonsai | [PyTorch(Author)](https://github.com/ldery/Bonsai)  | Language Modeling&Classification | 2024|
| 09 | [LaCo: Large Language Model Pruning via Layer Collapse](https://arxiv.org/abs/2402.11187) | arXiv | `L` | LaCo | - | Language Modeling&Classification | 2024|
| 10 | [ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853) | arXiv | `L` | ShortGPT | - | Language Modeling&Classification | 2024|
| 11 | [SparseLLM: Towards Global Pruning for Pre-trained Language Models](https://arxiv.org/abs/2402.17946) | arXiv | `B` | SparseLLM | [PyTorch(Author)](https://github.com/baithebest/sparsellm) | Language Modeling&Classification | 2024|
| 12 | [SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks](https://arxiv.org/abs/2402.09025) | arXiv | `N` | SLEB | [PyTorch(Author)](https://github.com/jiwonsong-dev/sleb) | Language Modeling&Classification | 2024|
| 13 | [Streamlining Redundant Layers to Compress Large Language Models](https://arxiv.org/abs/2403.19135) | arXiv | `L` | LLMStreamline | - | Language Modeling&Classification | 2024|
| 14 | [Why Lift so Heavy? Slimming Large Language Models by Cutting Off the Layers](https://arxiv.org/pdf/2402.11700) | arXiv | `L` | - | - |Classification | 2024|
| 15 | [Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods](https://arxiv.org/abs/2402.02834) |ICLRW | `HC` | - | [PyTorch(Author)](https://github.com/nota-netspresso/shortened-llm) |Classification | 2024|
| 16 | [Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity](https://arxiv.org/abs/2309.10285) | VLDB | `W` | Flash-LLM | [PyTorch(Author)](https://github.com/alibabaresearch/flash-llm) | Recognizing Textual Entailment | 2024|
| 17 | [The LLM Surgeon](https://arxiv.org/abs/2312.17244) | arXiv | `WC` | LLM Surgeon | [PyTorch(Author)](https://github.com/Qualcomm-AI-research/llm-surgeon) | Language Modeling | 2024|
| 18 | [Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity](https://arxiv.org/abs/2310.05175) | ICML | `W` | OWL | [PyTorch(Author)](https://github.com/luuyin/owl) | Language Modeling&Classification | 2024|
| 19 | [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887) | arXiv | `B` | - | - | Classification | 2024|
| 20 | [Enhancing One-Shot Pruned Generative Pre-training Language Models through Sparse-Dense-Sparse Mechanism](https://openreview.net/forum?id=TjXjkxhSdE&referrer=%5Bthe%20profile%20of%20Dong%20Li%5D(%2Fprofile%3Fid%3D~Dong_Li13)) | OpenReview | `W` | SDS | - | Classification | 2024|
| 21 | [KS-Lottery: Finding Certified Lottery Tickets for Multilingual Language Models](https://arxiv.org/abs/2402.02801) | arXiv | `W` | - | - | Language Translation | 2024|


#### 1.2.1 Data Augmentation
###### Pruning After Training LLMs 2023
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://arxiv.org/pdf/2301.00774.pdf) | NeurIPS |  `WP` | - | [PyTorch(Author)](https://github.com/IST-DASLab/sparsegpt)  | Language Modeling&Classification | 2023|
| 02 | [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627) | arXiv | `CHP` | LLM-Pruner |  [PyTorch(Author)](https://github.com/horseee/LLM-Pruner) | Language Modeling&Language Generation&Classification | 2023|
| 03 | [LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery](https://arxiv.org/abs/2310.18356) | arXiv | `CH` | LoRAShear | - | Language Modeling&Language Generation&Classification | 2023|
| 04 | [Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models](https://arxiv.org/abs/2310.05015) | arXiv | `CH` | Compresso | [PyTorch(Author)](https://github.com/microsoft/Moonlit/tree/main/Compresso) | Classification | 2023|
| 05 | [Mini-GPTs: Efficient Large Language Models through Contextual Pruning](https://arxiv.org/abs/2312.12682) | arXiv | `WC` | - | - |Language Modeling& Classification | 2023|
| 06 | [The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter](https://arxiv.org/abs/2306.03805) | arXiv |  `W`&N:M | - | [Pytorch(Author)](https://github.com/VITA-Group/essential_sparsity?tab=readme-ov-file) | NLP | 2023|


###### Pruning After Training Diffusion Models 2023
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Structural Pruning for Diffusion Models](https://arxiv.org/abs/2305.10924) |NeurIPS | `C` | Diff-Pruning | [PyTorch(Author)](https://github.com/VainF/Diff-Pruning) | Image Generation | 2023 |


###### Pruning After Training VLMs 2024
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models](https://arxiv.org/abs/2310.02998) | ICLR | `L` | ECoFLaP | [Pytorch(Author)](https://ecoflap.github.io/) | VQA&Image Captioning&Image-text Retrieval&Image Classification | 2024 |

###### Pruning After Training VLMs 2023
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Large Multimodal Model Compression via Efficient Pruning and Distillation at AntGroup](https://arxiv.org/abs/2312.05795) | arXiv | `B` | - | - | Multimodal Advertisement Audition | 2023 |
| 02 | [UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers](https://arxiv.org/abs/2301.13741) | ICML | `H` | UPop | [Pytorch(Author)](https://github.com/sdc17/UPop) | Image Classification&Image Caption&Image Retrieval&VQA | 2023 |
| 03 | [Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models](https://arxiv.org/abs/2306.10460) | ICML | `W` | ISP | [Pytorch(Author)](https://github.com/VITA-Group/instant_soup) | Image Classification&NLP | 2023 |

###### Pruning After Training VLMs 2022
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Playing Lottery Tickets with Vision and Language](https://arxiv.org/abs/2104.11832) | AAAI | `W` | - | - | Vision-and-Language | 2022 |


#### 1.2.2 Feature Augmentation
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://arxiv.org/abs/1905.09418) | ACL| `W` | - | [PyTorch(Author)](https://github.com/lena-voita/the-story-of-heads)| NLP | 2019 |
| 02 | [Playing the Lottery with Rewards and Multiple Languages: Lottery Tickets in RL and NLP](https://arxiv.org/abs/1906.02768) | ICLR | `W` | - | - | Classic Control&Atari Game | 2020 |
| 03 | [Dynamic Sparsity Neural Networks for Automatic Speech Recognition](https://arxiv.org/abs/2005.10627) | ICASSP | `W` |- | -| Speach Recognition | 2021 |
| 04 | [GAN Compression: Efficient Architectures for Interactive Conditional GANs](https://arxiv.org/pdf/2003.08936.pdf) | arXiv | `C` | - | - | Image-to-Image Translation | 2021 |
| 05 | [Content-Aware GAN Compression](https://arxiv.org/abs/2104.02244) | CVPR | `F` | - | [PyTorch(Author)](https://github.com/lychenyoko/content-aware-gan-compression) | Image Generation, Image Projection, Image Editing | 
| 06 | [A Unified Lottery Ticket Hypothesis for Graph Neural Networks](https://arxiv.org/abs/2102.06790) | ICML | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/Unified-LTH-GNN) | Node Classification&Link Prediction | 2021 |
| 07 | [Winning Lottery Tickets in Deep Generative Models](https://arxiv.org/abs/2010.02350) | AAAI | `W` | - | - | Image generative | 2021 |
| 08 | [GANs Can Play Lottery Tickets Too](https://arxiv.org/abs/2106.00134) | ICLR | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/GAN-LTH) | Image generative | 2021 |
| 09 | [Layer-wise Pruning of Transformer Attention Heads for Efficient Language Modeling](https://arxiv.org/abs/2110.03252) | arXiv | `H` | - | [PyTorch(Author)](https://github.com/aiha-lab/Attention-Head-Pruning) | Lanugage Modeling | 2021 |
| 10 | [Can We Find Strong Lottery Tickets in Generative Models?](https://arxiv.org/abs/2212.08311) | arXiv | `W` | - | - | Image generative | 2022 |
| 11 | [Exploring Lottery Ticket Hypothesis in Spiking Neural Networks](https://arxiv.org/abs/2207.01382) | ECCV | `W` | ET | [PyTorch(Author)](https://github.com/intelligent-computing-lab-yale/exploring-lottery-ticket-hypothesis-in-snns) | Image Classification | 2022 |
| 12 | [Structured Pruning for Efficient Generative Pre-trained Language Models](https://aclanthology.org/2023.findings-acl.692/) | ACL | `C` | CP3 | - | Language Modeling&Machine Translation&Abstractive Summarization | 2023 |
| 13 | [Rethinking Graph Lottery Tickets: Graph Sparsity Matters](https://arxiv.org/abs/2305.02190) | ICLR | `W` | - | - | Node Classification | 2023 |
| 14 | [CP3: Channel Pruning Plug-in for Point-based Networks](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_CP3_Channel_Pruning_Plug-In_for_Point-Based_Networks_CVPR_2023_paper.pdf) | CVPR | `C` | CP3 | - | 3D Image Classification and Object Detection | 2023 |


###### Post Training 2024
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Fast and Controllable Post-training Sparsity: Learning Optimal Sparsity Allocation with Global Constraint in Minutes](https://arxiv.org/abs/2203.04570) | AAAI | `W` | FCPTS | - | Image Classification | 2024 |

###### Post Training 2023
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://arxiv.org/pdf/2301.00774.pdf) | NeurIPS |  `WP` | - | [PyTorch(Author)](https://github.com/IST-DASLab/sparsegpt)  | Language Modeling | 2023|
| 02 | [Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning](https://arxiv.org/abs/2308.07209) | ICCV |  `C` | UDFC | -  | Image Classification | 2023|
| 03 | [OTOv3: Automatic Architecture-Agnostic Neural Network Training and Compression from Structured Pruning to Erasing Operators](https://arxiv.org/abs/2312.09411) | arXiv | `WFC`  | - | -  | Image Classification | 2023|

###### Post Training 2022
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [CP-ViT: Cascade Vision Transformer Pruning via Progressive Sparsity Prediction](https://arxiv.org/abs/2203.04570) | arXiv | `H` | CP-ViT  | - | Image Classification | 2022 |
| 02 | [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://arxiv.org/abs/2208.11580) | NeurIPS | `W` | ExactOBS  | [PyTorch(Author)](https://github.com/IST-DASLab/OBC) | Image Classification&Object Detection&Question Answering | 2022 |
| 03 | [A Fast Post-Training Pruning Framework for Transformers](https://arxiv.org/pdf/2210.04092.pdf) | NeurIPS |  `HF` | - | [PyTorch(Author)](https://github.com/WoosukKwon/retraining-free-pruning)  | Natural Language Understanding | 2022|

###### Post Training 2021
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Enabling Retrain-free Deep Neural Network Pruning Using Surrogate Lagrangian Relaxation](https://arxiv.org/abs/2012.10079) | IJCAI | `W` | - | - | Image Classification & Object Detection | 2021 |
| 02 | [Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks](https://arxiv.org/abs/2102.08124)  | NeurIPS | N:M | AdaPrune | [PyTorch(Author)](https://github.com/papers-submission/structured_transposable_masks) | Image Classification | 2021 | 


###### Post Training 2021
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Linear Mode Connectivity and the Lottery Ticket Hypothesis](https://arxiv.org/abs/1912.05671) | ICML | `W` | - | - | Image Classification | 2020 |
| 02 | [When To Prune? A Policy Towards Early Structural Pruning](https://openaccess.thecvf.com/content/CVPR2022/html/Shen_When_To_Prune_A_Policy_Towards_Early_Structural_Pruning_CVPR_2022_paper.html) | CVPR | `F` | PaT | - | Image Classification | 2022 |
| 03 | [Drawing Early-Bird Tickets: Towards More Efficient Training of Deep Networks](https://arxiv.org/abs/1909.11957) | ICLR | `W` | - | [PyTorch(Author)](https://github.com/GATECH-EIC/Early-Bird-Tickets) | Image Classification | 2020 |
| 04 | [A Gradient Flow Framework For Analyzing Network Pruning](https://arxiv.org/abs/2009.11839) | ICLR | `F` | - | [PyTorch(Author)](https://github.com/ModelTC/FCPTS) | Image Classification | 2021 |

###### Post Training 2021
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Channel Gating Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2017/file/a51fb975227d6640e4fe47854476d133-Paper.pdf) | NeurIPS | `F` | RNP | - | Image Classification | 2017 |
| 02 | [Channel Gating Neural Networks](https://arxiv.org/abs/1805.12549) | NeurIPS | `C` | CGNet | [PyTorch(Author)](https://github.com/cornell-zhang/dnn-gating) | Image Classification | 2019 |
| 03 | [Dynamic Channel Pruning: Feature Boosting and Suppression](https://arxiv.org/pdf/1810.05331.pdf) | ICLR | `C` | FBS | [PyTorch(Author)](https://github.com/YOUSIKI/PyTorch-FBS) | Image Classification | 2019 |
| 04 | [Frequency-Domain Dynamic Pruning for Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2018/file/a9a6653e48976138166de32772b1bf40-Paper.pdf) | NeurIPS | `F` | FDNP | - | Image Classification | 2019 |
| 05 | [Fire Together Wire Together: A Dynamic Pruning Approach With Self-Supervised Mask Prediction](https://openaccess.thecvf.com/content/CVPR2022/html/Elkerdawy_Fire_Together_Wire_Together_A_Dynamic_Pruning_Approach_With_Self-Supervised_CVPR_2022_paper.html) | CVPR| `F` | - | - | Image Classification | 2019 |
| 06 | [Dynamic Dual Gating Neural Networks](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Dynamic_Dual_Gating_Neural_Networks_ICCV_2021_paper.pdf) | ICCV | `C` | DGNet | [PyTorch(Author)](https://github.com/lfr-0531/DGNet) | Image Classification | 2021 |
| 07 | [Manifold Regularized Dynamic Network Pruning](https://arxiv.org/abs/2103.05861) | CVPR | `F` | ManiDP |  [PyTorch(Author)](https://github.com/huaweinoah/Pruning/tree/master/ManiDP) | Image Classification | 2021 |
| 08 | [Contrastive Dual Gating: Learning Sparse Features With Contrastive Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Meng_Contrastive_Dual_Gating_Learning_Sparse_Features_With_Contrastive_Learning_CVPR_2022_paper.html) | CVPR | `WF` | CDG | - | Image Classification | 2022 |


## 2. model Based Approaches

### 2.1 Architecture-Based Approches

#### 2.1.1 Attention-Based
| No. | Title   | Venue | Algorithm Name | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|:----:|
| 01 | [Continual Learning via Neural Pruning](https://arxiv.org/abs/1903.04476)| arXiv | CLNP | - | Image Classification | 2019 |
| 02 | [Learning Bayesian Sparse Networks With Full Experience Replay for Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Yan_Learning_Bayesian_Sparse_Networks_With_Full_Experience_Replay_for_Continual_CVPR_2022_paper.html)| CVPR | SNCL | - | Image Classification | 2022 |  
| 03 | [Continual Prune-and-Select: Class-Incremental Learning with SPecialized Subnetworks](https://arxiv.org/pdf/2208.04952.pdf)| Applied Intelligence | - | [PyTorch(Author)]( https://github.com/adekhovich/continual_prune_and_select) | Image Classification | 2023 |
| 04 | [Continual Domain Adaptation through Pruning-aided Domain-specific Weight Modulation](https://arxiv.org/abs/2304.07560)| CVPRW | PaCDA | [PyTorch(Author)]( https://github.com/prasannab29/pacda) | Image Classification | 2023 |

#### 2.1.2 Dynamic network structure-based
| No. | Title   | Venue | Algorithm Name | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|:----:|
| 01 | [Studying the impact of magnitude pruning on contrastive learning methods](https://arxiv.org/pdf/2207.00200.pdf) | ICML | - | [PyTorch(Author)](https://github.com/FraCorti/Studying-the-impact-of-magnitude-pruning-on-contrastive-learning-methods) | Image Classification | 2020 |
| 02 | [Training Debiased Subnetworks with Contrastive Weight Pruning](https://openaccess.thecvf.com/content/CVPR2023/papers/Park_Training_Debiased_Subnetworks_With_Contrastive_Weight_Pruning_CVPR_2023_paper.pdf) | CVPR | DCWP | - | Image Classification | 2023 |

###### Post Training 2021
| No. | Title   | Venue | Algorithm Name | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|:----:|
| 01 | [FedDUAP: Federated Learning with Dynamic Update and Adaptive Pruning Using Shared Data on the Server](https://arxiv.org/pdf/2204.11536.pdf) | IJCAI | FedDUAP | - | Image Classification | 2020 |
| 02 | [Model Pruning Enables Efficient Federated Learning on Edge Devices](https://arxiv.org/pdf/1909.12326.pdf) | TNNLS | - | [PyTorch(Author)](https://github.com/jiangyuang/PruneFL) | Image Classification | 2022 |


## 3. Optimization Based Approaches

### 3.1 Gradient-Based Approaches
#### 3.1.1 Function Regularization
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Deep Rewiring: Training very Sparse Deep Networks](https://arxiv.org/pdf/1711.05136.pdf) | ICLR | - | Image Classification&Audio | 2018 |
| 02 | [Co-Evolutionary Compression for Unpaired Image Translation](https://arxiv.org/pdf/1907.10804.pdf) | ICCV | [PyTorch(Author)](https://github.com/yehuitang/Pruning) | Image Style Translation | 2019 |
| 03 | [Content-Aware GAN Compression](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Content-Aware_GAN_Compression_CVPR_2021_paper.pdf) | CVPR |  [PyTorch(Author)](https://github.com/lychenyoko/content-aware-gan-compression) | Image Style Translation | 2021 |
| 04 | [Training Neural Networks with Fixed Sparse Masks](https://arxiv.org/abs/2111.09839) | NeurIPS | [PyTorch(Author)]( https://github.com/varunnair18/FISH) | Image Classification | 2021 |
| 05 | [Vision Transformer Slimming: Multi-Dimension Searching in Continuous Optimization Space](https://openaccess.thecvf.com/content/CVPR2022/papers/Chavan_Vision_Transformer_Slimming_Multi-Dimension_Searching_in_Continuous_Optimization_Space_CVPR_2022_paper.pdf) | CVPR | [PyTorch(Author)](https://github.com/Arnav0400/ViT-Slim) | Image Classification&Audio | 2022 |
| 06 | [SuperTickets: Drawing Task-Agnostic Lottery Tickets from Supernets via Jointly Architecture Searching and Parameter Pruning](https://arxiv.org/abs/2207.03677) | ECCV | [PyTorch(Author)](https://github.com/GATECH-EIC/SuperTickets) | Image Classification&Object Detection&Human Pose Estimation | 2022 |


#### 3.1.2 Weight Regularization

| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [When BERT Plays the Lottery, All Tickets Are Winning](https://arxiv.org/abs/2005.00561) | EMNLP | [PyTorch(Author)](https://github.com/sai-prasanna/bert-experiments) | Language Modeling | 2020 |
| 02 | [The Lottery Ticket Hypothesis for Pre-trained BERT Networks](https://arxiv.org/abs/2007.12223) | ICML | [PyTorch(Author)](https://github.com/VITA-Group/BERT-Tickets) | Language Modeling | 2021 |
| 03 | [Structured Pruning Learns Compact and Accurate Models](https://arxiv.org/pdf/2204.00408.pdf) | ACL | [PyTorch(Author)](https://github.com/OPTML-Group/BiP)  | Natural Language Understanding | 2022|
| 04 | [A Fast Post-Training Pruning Framework for Transformers](https://arxiv.org/pdf/2204.09656.pdf) | NeurIPS | [PyTorch(Author)](https://github.com/WoosukKwon/retraining-free-pruning) | Natural Language Understanding | 2022 |
| 05 | [A Fast Post-Training Pruning Framework for Transformers](https://arxiv.org/pdf/2210.04092.pdf) | NeurIPS | [PyTorch(Author)](https://github.com/WoosukKwon/retraining-free-pruning)  | Natural Language Understanding | 2022|
| 06 | [The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models](https://arxiv.org/pdf/2203.07259.pdf) | EMNLP | [PyTorch(Author)](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT)| Natural Language Understanding | 2022|
| 07 | [Pruning Meets Low-Rank Parameter-efficient](https://arxiv.org/abs/2305.18403) | arXiv |  -  | Image Classification&Language Modeling | 2023|
| 08 | [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627) | arXiv |  -  | Language Modeling | 2023|

#### 3.1.3 Gradient Space
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Exploring Sparsity in recurrent neural networks](https://arxiv.org/abs/1704.05119) | ICLR | [PyTorch](https://github.com/puhsu/pruning) | Speech Recognition | 2017 |
| 02 | [Deep Rewiring: Training very Sparse Deep Networks](https://arxiv.org/pdf/1711.05136.pdf) | ICLR | - | Image Classification&Audio | 2018 |


#### 3.1.4 Loss Function
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Exploring Sparsity in recurrent neural networks](https://arxiv.org/abs/1704.05119) | ICLR | [PyTorch](https://github.com/puhsu/pruning) | Speech Recognition | 2017 |
| 02 | [Deep Rewiring: Training very Sparse Deep Networks](https://arxiv.org/pdf/1711.05136.pdf) | ICLR | - | Image Classification&Audio | 2018 |

### 3.2 Tuning-Based Approaches
#### 3.2.1 Adapter-Based
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization](https://openaccess.thecvf.com/content_cvpr_2018/html/Tung_CLIP-Q_Deep_Network_CVPR_2018_paper.html)  | CVPR | - | Image Classification | 2018 |
| 02 | [Accelerating Sparse Deep Neural Networks](https://arxiv.org/pdf/2104.08378.pdf) | arXiv | - | Image Classification&Object Detection&Language Translation&Language Modeling&Image Synthesis&Domain Translation&Style Transfer&Image-Image Translation&Super Resolution | 2021 |
| 03 | [OPQ: Compressing Deep Neural Networks with One-shot Pruning-Quantization](https://arxiv.org/pdf/2205.11141.pdf) | AAAI | - | Image Classification | 2021 |
| 04 | [Deep Model Compression Based on the Training History](https://arxiv.org/pdf/2102.00160.pdf) | arXiv | - | Image Classification | 2022 |
| 05 | [LLM-Pruner: On the Structural Pruning of Large Language Models](arxiv.org/abs/2305.11627) | arXiv | [PyTorch](https://github.com/horseee/LLM-Pruner) | Causal Language Modeling | 2023 |
| 06 | [Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning](https://openaccess.thecvf.com/content/ICCV2023/papers/Bai_Unified_Data-Free_Compression_Pruning_and_Quantization_without_Fine-Tuning_ICCV_2023_paper.pdf) | ICCV | - | Image Classification | 2023 |

#### 3.2.2 Instruct Tuning
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization](https://openaccess.thecvf.com/content_cvpr_2018/html/Tung_CLIP-Q_Deep_Network_CVPR_2018_paper.html)  | CVPR | - | Image Classification | 2018 |
| 02 | [Accelerating Sparse Deep Neural Networks](https://arxiv.org/pdf/2104.08378.pdf) | arXiv | - | Image Classification&Object Detection&Language Translation&Language Modeling&Image Synthesis&Domain Translation&Style Transfer&Image-Image Translation&Super Resolution | 2021 |
| 03 | [OPQ: Compressing Deep Neural Networks with One-shot Pruning-Quantization](https://arxiv.org/pdf/2205.11141.pdf) | AAAI | - | Image Classification | 2021 |
| 04 | [Deep Model Compression Based on the Training History](https://arxiv.org/pdf/2102.00160.pdf) | arXiv | - | Image Classification | 2022 |
| 05 | [LLM-Pruner: On the Structural Pruning of Large Language Models](arxiv.org/abs/2305.11627) | arXiv | [PyTorch](https://github.com/horseee/LLM-Pruner) | Causal Language Modeling | 2023 |
| 06 | [Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning](https://openaccess.thecvf.com/content/ICCV2023/papers/Bai_Unified_Data-Free_Compression_Pruning_and_Quantization_without_Fine-Tuning_ICCV_2023_paper.pdf) | ICCV | - | Image Classification | 2023 |

#### 3.2.3 Prompt-Based
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization](https://openaccess.thecvf.com/content_cvpr_2018/html/Tung_CLIP-Q_Deep_Network_CVPR_2018_paper.html)  | CVPR | - | Image Classification | 2018 |
| 02 | [Accelerating Sparse Deep Neural Networks](https://arxiv.org/pdf/2104.08378.pdf) | arXiv | - | Image Classification&Object Detection&Language Translation&Language Modeling&Image Synthesis&Domain Translation&Style Transfer&Image-Image Translation&Super Resolution | 2021 |
| 03 | [OPQ: Compressing Deep Neural Networks with One-shot Pruning-Quantization](https://arxiv.org/pdf/2205.11141.pdf) | AAAI | - | Image Classification | 2021 |
| 04 | [Deep Model Compression Based on the Training History](https://arxiv.org/pdf/2102.00160.pdf) | arXiv | - | Image Classification | 2022 |
| 05 | [LLM-Pruner: On the Structural Pruning of Large Language Models](arxiv.org/abs/2305.11627) | arXiv | [PyTorch](https://github.com/horseee/LLM-Pruner) | Causal Language Modeling | 2023 |
| 06 | [Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning](https://openaccess.thecvf.com/content/ICCV2023/papers/Bai_Unified_Data-Free_Compression_Pruning_and_Quantization_without_Fine-Tuning_ICCV_2023_paper.pdf) | ICCV | - | Image Classification | 2023 |


#### 3.2.4 Prefix-Tuning
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization](https://openaccess.thecvf.com/content_cvpr_2018/html/Tung_CLIP-Q_Deep_Network_CVPR_2018_paper.html)  | CVPR | - | Image Classification | 2018 |
| 02 | [Accelerating Sparse Deep Neural Networks](https://arxiv.org/pdf/2104.08378.pdf) | arXiv | - | Image Classification&Object Detection&Language Translation&Language Modeling&Image Synthesis&Domain Translation&Style Transfer&Image-Image Translation&Super Resolution | 2021 |
| 03 | [OPQ: Compressing Deep Neural Networks with One-shot Pruning-Quantization](https://arxiv.org/pdf/2205.11141.pdf) | AAAI | - | Image Classification | 2021 |
| 04 | [Deep Model Compression Based on the Training History](https://arxiv.org/pdf/2102.00160.pdf) | arXiv | - | Image Classification | 2022 |
| 05 | [LLM-Pruner: On the Structural Pruning of Large Language Models](arxiv.org/abs/2305.11627) | arXiv | [PyTorch](https://github.com/horseee/LLM-Pruner) | Causal Language Modeling | 2023 |
| 06 | [Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning](https://openaccess.thecvf.com/content/ICCV2023/papers/Bai_Unified_Data-Free_Compression_Pruning_and_Quantization_without_Fine-Tuning_ICCV_2023_paper.pdf) | ICCV | - | Image Classification | 2023 |


## 4. Survey of Pruning
### Survey of Pruning 2024
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Structured Pruning for Deep Convolutional Neural Networks: A survey](https://arxiv.org/pdf/2303.00566.pdf) | TPAMI | - | CV&NLP | 2024 |
| 02 | [A survey on efficient vision transformers: algorithms, techniques, and performance benchmarking](https://arxiv.org/abs/2309.02031) | arXiv | - | CV | 2024 |
| 03 | [A Survey of Lottery Ticket Hypothesis](https://arxiv.org/abs/2403.04861) | arXiv | - | CV&NLP | 2024 |
| 04 | [Model Compression and Efficient Inference for Large Language Models: A Survey](https://arxiv.org/abs/2402.09748) | arXiv | - | NLP | 2024 |

### Survey of Pruning 2023
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning](https://arxiv.org/pdf/2301.05219.pdf) | arXiv | [PyTorch(Author)](https://github.com/MingSun-Tse/Why-the-State-of-Pruning-So-Confusing) | Image Classification | 2023 |
| 02 | [Transforming Large-Size to Lightweight Deep Neural Networks for IoT Applications](https://dl.acm.org/doi/10.1145/3570955) | ACM Computing Surveys | - | CV&NLP&Audio | 2023 |
| 03 | [A Survey on Model Compression for Large Language Models](https://arxiv.org/abs/2308.07633) | TACL | - | NLP&Unseen Instructions | 2023 |
| 04 | [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234) | arXiv | - | - | 2023 |
| 05 | [A Survey on Dynamic Neural Networks for Natural Language Processing](https://arxiv.org/pdf/2202.07101.pdf) | arXiv | - | NLP | 2023 |
| 06 | [Dimensionality Reduced Training by Pruning and Freezing Parts of a Deep Neural Network, a Survey](https://arxiv.org/abs/2205.08099) | arXiv | - | CV&NLP | 2023 |


### Survey of Pruning 2022
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [A Survey on Efficient Convolutional Neural Networks and Hardware Acceleration](https://arxiv.org/pdf/2103.06460.pdf) | Electronics | - | - | 2022 |
| 02 | [Dimensionality Reduced Training by Pruning and Freezing Parts of a Deep Neural Network, a Survey](https://arxiv.org/pdf/2205.08099.pdf) | arXiv | - | Image Classification | 2022 |
| 03 | [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) | arXiv | - | CV&NLP | 2022 |
| 04 | [Recent Advances on Neural Network Pruning at Initialization](https://arxiv.org/pdf/2103.06460.pdf) | IJCAI | - | CV&NLP | 2022 |

### Survey of Pruning 2021
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks](https://arxiv.org/abs/2102.00554) | JMLR | - | Image Classification | 2021 |
| 02 | [Dynamic Neural Networks: A Survey](https://arxiv.org/pdf/2102.04906.pdf) | arXiv | - | - | 2021 |
| 03 | [Pruning and Quantization for Deep Neural Network Acceleration: A Survey](https://arxiv.org/pdf/2101.09671.pdf) | Neurocomputing | - | Image Classification | 2021 |
| 04 | [Compressing Large-Scale Transformer-Based Models: A Case Study on BERT](https://arxiv.org/abs/2002.11985) | TACL | - | NLP | 2021 |

### Survey of Pruning 2020
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Model Compression and Hardware Acceleration for Neural Networks: A Comprehensive Survey](https://ieeexplore.ieee.org/document/9043731) | IEEE | - | - | 2020 |
| 02 | [Pruning Algorithms to Accelerate Convolutional Neural Networks for Edge Applications: A Survey](https://arxiv.org/pdf/2005.04275.pdf) | arXiv | - | Image Classification | 2020 |
| 03 | [A Survey of Model Compression and Acceleration for Deep Neural Networks](https://arxiv.org/pdf/1710.09282.pdf) | arXiv | - | - | 2020 |
| 04 | [An Survey of Neural Network Compression](https://arxiv.org/pdf/2006.03669.pdf) | arXiv | - | - | 2020 |
| 05 | [Convolutional Neural Network Pruning: A Survey](https://ieeexplore.ieee.org/document/9189610) | CCC | - | - | 2020 |
| 06 | [What is the State of Neural Network Pruning?](https://arxiv.org/pdf/2003.03033.pdf) | MLSys | - | - | 2020 |
| 07 | [A comprehensive survey on model compression and acceleration](https://link.springer.com/article/10.1007/s10462-020-09816-7) | Artificial Intelligence Review | - | - | 2020 |
| 08 | [A Survey on Deep Neural Network Compression: Challenges, Overview, and Solutions](https://arxiv.org/pdf/2010.03954.pdf) | arXiv | - | - | 2020 |

### Survey of Pruning 2019 and earlier
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Pruning Algorithms-A Survey](https://ieeexplore.ieee.org/document/248452) | IEEE Transactions on Neural Networks | - | Image Classification | 1993 |
| 02 | [Efficient Processing of Deep Neural Networks: A Tutorial and Survey](https://arxiv.org/abs/1703.09039) | arXiv | - | Image Classification | 2017 |
| 03 | [Recent advances in efficient computation of deep convolutional neural networks](https://arxiv.org/pdf/1802.00939.pdf) | arXiv | - | - | 2018 |
| 04 | [The State of Sparsity in Deep Neural Networks](https://arxiv.org/abs/1902.09574) | arXiv | [PyTorch(Author)](https://github.com/google-research/google-research/blob/master/state_of_sparsity/README.md) | Image Classification&machine translation | 2019 |


## 5. Other Works
### Papers
| No. | Title   | Venue | Algorithm Name | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:-----:|:-------:|:----:|:----:|:----:|
| 01 | [Is Pruning Compression?: Investigating Pruning Via Network Layer Similarity](https://openaccess.thecvf.com/content_WACV_2020/papers/Blakeney_Is_Pruning_Compression_Investigating_Pruning_Via_Network_Layer_Similarity_WACV_2020_paper.pdf) | WACV | - | - | Image Classification | 2020 |
| 02 | [A Gradient Flow Framework For Analyzing Network Pruning](https://openreview.net/forum?id=rumv7QmLUue) | ICLR | - | [PyTorch(Author)](https://github.com/EkdeepSLubana/flowandprune) | Image Classification | 2021 |
| 03 | [Data Level Lottery Ticket Hypothesis for Vision Transformers](https://arxiv.org/abs/2211.01484) | IJCAI | - | [PyTorch(Author)](https://github.com/shawnricecake/vit-lottery-ticket-input) | Image Classification | 2021 |
| 04 | [Are All Layers Created Equal?](https://arxiv.org/abs/1902.01996) | JMLR | - | - | Image Classification | 2022 |


