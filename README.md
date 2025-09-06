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
 
 
#### 1.1.4 Pseudo-scenarios Reply
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


#### 1.1.5 Raw-data replay
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
###### Data Augmentation 2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Cascade prompt learning for vision-language model adaptation](https://arxiv.org/pdf/2409.17805) | ECCV  | CasPL  | [GitHub](https://github.com/megvii-research/CasPL) | 2024 |
| 02 |  [Concept-guided prompt learning for generalization in vision-language models](https://ojs.aaai.org/index.php/AAAI/article/download/28568/29104)  | AAAI | CPL   | [GitHub](https://github.com/rambo-coder/CPL) | 2024 |
| 03 |  [Making pre-trained language models better continual few-shot relation extractors](https://arxiv.org/pdf/2402.15713)   | arXiv | CPL   | [GitHub](https://github.com/mashengkun/CPL) | 2024 |
| 04 |   [Conditional prototype rectification prompt learning](https://ieeexplore.ieee.org/abstract/document/11069290/)  | TCSVT   | CPR  | [GitHub](https://github.com/chenhaoxing/CPR) | 2024 |
| 05 |  [M³PL: Identifying and Exploiting View Bias of Prompt Learning](https://openreview.net/pdf?id=2rnTIBm19V) | Transact Mach Learn Res | M³PL | - | 2024 |
| 06 | [Rethinking misalignment in vision-language model adaptation from a causal perspective](https://proceedings.neurips.cc/paper_files/paper/2024/file/453a27b717972ef94a9a9113d236ad2f-Paper-Conference.pdf) | NeurIPS | CDC | [NeurIPS Code Policy](https://nips.cc/public/guides/CodeSubmissionPolicy) | 2024 |
| 07 |  [Learning to learn for few-shot continual active learning](https://link.springer.com/content/pdf/10.1007/s10462-024-10924-x.pdf) | ARTIF INTELL REV | Meta-CAL | [PyTorch 1.10.0+cu113](https://pytorch.org/1.10.0+cu113) | 2024 |
| 08 |  [HPT++: Hierarchically Prompting Vision-Language Models with Multi-Granularity Knowledge Generation and Improved Structure Modeling](https://arxiv.org/pdf/2408.14812) | arXiv | HPT++ | - | 2024 |
| 09 |  [Learning hierarchical prompt with structured linguistic knowledge for vision-language models](https://ojs.aaai.org/index.php/AAAI/article/view/28387/28756) | AAAI | HPT | [GitHub](https://github.com/Vill-Lab/2024-AAAI-HPT) | 2024 |
| 10 |   [Active prompt learning in vision language models](https://openaccess.thecvf.com/content/CVPR2024/papers/Bang_Active_Prompt_Learning_in_Vision_Language_Models_CVPR_2024_paper.pdf) | CVPR | PCB | [GitHub](https://github.com/kaist-dmlab/pcb) | 2024 |


###### Data Augmentation 2023
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Serial contrastive knowledge distillation for continual few-shot relation extraction](https://arxiv.org/pdf/2305.06616) | arXiv | SCKD | [GitHub](https://github.com/nju-websoft/SCKD) | 2023 |
| 02 | [Task-oriented multi-modal mutual leaning for vision-language models](https://openaccess.thecvf.com/content/ICCV2023/papers/Long_Task-Oriented_Multi-Modal_Mutual_Leaning_for_Vision-Language_Models_ICCV_2023_paper.pdf) | ICCV | CTP | - | 2023 |
| 03 |[LoGoPrompt: Synthetic text images can be good visual prompts for vision-language models](https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_LoGoPrompt_Synthetic_Text_Images_Can_Be_Good_Visual_Prompts_for_ICCV_2023_paper.pdf) | ICCV | LoGoPrompt | [GitHub](https://chengshiest.github.io/logo) | 2023 |
| 04 |[MaPLe: Multi-modal prompt learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Khattak_MaPLe_Multi-Modal_Prompt_Learning_CVPR_2023_paper.pdf) | CVPR | MaPLe | [GitHub](https://github.com/muzairkhattak/multimodal-prompt-learning) | 2023 |
| 05 |[Ranpac: Random projections and pre-trained models for continual learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/2793dc35e14003dd367684d93d236847-Paper-Conference.pdf) | NeurIPS | RanPAC | [GitHub](https://github.com/RanPAC/RanPAC) | 2023 |
| 06 |[Few shot class incremental learning via efficient prototype replay and calibration](https://www.mdpi.com/1099-4300/25/5/776) | Entropy | EPRC | - | 2023 |

###### Data Augmentation 2022 and earlier
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Learning to decompose visual features with latent textual prompts](https://arxiv.org/pdf/2210.04287) | arXiv | DeFo | - | 2022 |
| 02 |[Plot: Prompt learning with optimal transport for vision-language models](https://arxiv.org/pdf/2210.01253) | arXiv | PLOT | [GitHub](https://github.com/CHENGY12/PLOT) | 2022 |
| 03 | [Prompt distribution learning](http://openaccess.thecvf.com/content/CVPR2022/papers/Lu_Prompt_Distribution_Learning_CVPR_2022_paper.pdf) | CVPR | ProDA | - | 2022 |
| 04 |[Conditional prompt learning for vision-language models](http://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.pdf) | CVPR | CoCoOp | [GitHub](https://github.com/KaiyangZhou/CoOp) | 2022 |
| 05 |[Learning to prompt for vision-language models](https://arxiv.org/pdf/2109.01134) | IJCV | CoOp | [GitHub](https://github.com/KaiyangZhou/CoOp) | 2022 |
| 06 |[Continual few-shot relation learning via embedding space regularization and data augmentation](https://arxiv.org/pdf/2203.02135) | arXiv | CFRL | [GitHub](https://github.com/qcwthu/Continual_Fewshot_Relation_Learning) | 2022 |
| 07 | zhu2021self | [Self-promoted prototype refinement for few-shot class-incremental learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Self-Promoted_Prototype_Refinement_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf) | CVPR | SPPR | - | 2021 |

#### 1.2.2 Feature Augmentation
###### Feature Augmentation 2025
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
|01| [Contrastive augmented graph2graph memory interaction for few shot continual learning](https://ieeexplore.ieee.org/abstract/document/10841449/ ) | TCSVT | LGP | -  | 2025 |


###### Feature Augmentation 2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
|01| [Improving Continual Few-shot Relation Extraction through Relational Knowledge Distillation and Prototype Augmentation](https://aclanthology.org/2024.lrec-main.767.pdf) | LREC-COLING  | RK2DA | - | 2024 |
|02| [Making pre-trained language models better continual few-shot relation extractors](https://arxiv.org/pdf/2402.15713) | arXiv| CPL | [GitHub](https://github.com/mashengkun/CPL)| 2024 |
|03| [Conditional prototype rectification prompt learning](https://ieeexplore.ieee.org/abstract/document/11069290/) | TCSVT  | CPR  | [GitHub](https://github.com/chenhaoxing/CPR)  | 2024 |
|04| [Modal-aware prompt tuning with deep adaptive feature enhancement](https://www.sciencedirect.com/science/article/abs/pii/S0045790624001988)| COMPUT_ELECTR_ENG | MAP-DAFE| -| 2024 |
|05| [A hard-to-beat baseline for training-free clip-based adaptation](https://arxiv.org/pdf/2402.04087) | arXiv | GDA  | [GitHub](https://github.com/mrflogs/ICLR24) | 2024 |
|06| [APLe: Token-Wise Adaptive for Multi-Modal Prompt Learning] (https://arxiv.org/pdf/2401.06827) | arXiv | APLe | - | 2024 |


###### Feature Augmentation 2023
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Consistent prototype learning for few-shot continual relation extraction](https://aclanthology.org/2023.acl-long.409.pdf ) | ACL | ConPL  | [GitHub](https://github.com/XiudiChen/ConPL) | 2023 |
| 02  | [Ica-proto: Iterative cross alignment prototypical network for incremental few-shot relation classification](https://aclanthology.org/2023.findings-eacl.171.pdf) | EACL | ICA-Proto  | -  | 2023 |
| 03  | [Task-oriented multi-modal mutual leaning for vision-language models](https://openaccess.thecvf.com/content/ICCV2023/papers/Long_Task-Oriented_Multi-Modal_Mutual_Leaning_for_Vision-Language_Models_ICCV_2023_paper.pdf)    | ICCV | CTP  | -  | 2023 |
| 04  | [Ranpac: Random projections and pre-trained models for continual learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/2793dc35e14003dd367684d93d236847-Paper-Conference.pdf) | NeurIPS  | RanPAC  | [GitHub](https://github.com/RanPAC/RanPAC)  | 2023 |
| 05  | [Few-shot class-incremental learning for medical time series classification](https://drive.google.com/file/d/1hsaJUvUPMAcMHuAoqL0wKUssG_o9rR71/view) | J-BHI   | MAPIC | - | 2023 |
| 06  | [Gkeal: Gaussian kernel embedded analytic learning for few-shot class incremental task](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhuang_GKEAL_Gaussian_Kernel_Embedded_Analytic_Learning_for_Few-Shot_Class_Incremental_CVPR_2023_paper.pdf) | CVPR | GKEAL  | -  | 2023 |
| 07 | [Self-regulating prompts: Foundational model adaptation without forgetting](https://openaccess.thecvf.com/content/ICCV2023/papers/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.pdf ) | ICCV | PromptSR | [GitHub](https://github.com/muzairkhattak/PromptSRC) | 2023 |

###### Feature Augmentation 2022 and earlier
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Self-promoted prototype refinement for few-shot class-incremental learning]( https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Self-Promoted_Prototype_Refinement_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf) | CVPR   | SPPR | -| 2021 |
| 02  | [Prototype augmentation and self-supervision for incremental learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf)|CVPR | PASS| [GitHub](https://github.com/Impression2805/CVPR21_PASS)| 2021 |
| 03  | [Memory-efficient incremental learning through feature adaptation](https://link.springer.com/chapter/10.1007/978-3-030-58517-4_41 )| ECCV |Feature_Adaptation | -  | 2020 |

## 2. Model Based Approaches 
### 2.1 Architecture-Based Approaches 
#### 2.1.1 Attention-based
###### Attention-based 2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [SEP: Self-Enhanced Prompt Tuning for Visual-Language Model](https://arxiv.org/pdf/2405.15549) |arXiv| SEP| [GitHub](https://github.com/htyao89/SEP/) | 2024 |
| 02  | [Semantic Alignment for Prompt-Tuning in Vision Language Models](https://openreview.net/pdf?id=avDr56QjSI) | TMLR  | SAP | -  | 2024 |
| 03  | [Revisiting prompt pretraining of vision-language models](https://arxiv.org/pdf/2409.06166) | arXiv | RPP | - | 2024 |
| 04 | [Few-shot class incremental learning with attention-aware self-adaptive prompt](https://link.springer.com/chapter/10.1007/978-3-031-73004-7_1) | ECCV         | ASP | [GitHub](https://github.com/DawnLIU35/FSCIL-ASP) | 2024 |
|05| [M2sd: Multiple mixing self-distillation for few-shot class-incremental learning](https://ojs.aaai.org/index.php/AAAI/article/view/28129) | AAAI| M2SD | - |2024|
|06| [HPT++: Hierarchically Prompting Vision-Language Models with Multi-Granularity Knowledge Generation and Improved Structure Modeling](https://arxiv.org/pdf/2408.14812) | arXiv | HPT++  | - | 2024 |

###### Attention-based 2023 and earlier
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
|01| [Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) | CVPR | CODA-Prompt | [GitHub](https://github.com/GT-RIPL/CODA-Prompt)  | 2023 |
|02| [Dpl: Decoupled prompt learning for vision-language models](https://arxiv.org/pdf/2308.10061) | arXiv | DPL | -   | 2023 |
|03| [Continual learning with lifelong vision transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Continual_Learning_With_Lifelong_Vision_Transformer_CVPR_2022_paper.pdf) | CVPR  | LVT  | - | 2022 |

#### 2.1.2 Dynamic network structure-based
###### Dynamic network structure-based 2025
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
|01|[Skip tuning: Pre-trained vision-language models are effective and efficient adapters themselves](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Skip_Tuning_Pre-trained_Vision-Language_Models_are_Effective_and_Efficient_Adapters_CVPR_2025_paper.pdf) | CVPR | Skip_Tuning| [GitHub](https://github.com/Koorye/SkipTuning)| 2025 |
|02| [Continuous Knowledge-Preserving Decomposition for Few-Shot Continual Learning](https://arxiv.org/pdf/2501.05017) | arXiv | CKPD-FSCIL| [GitHub](https://github.com/xiaojieli0903/CKPD-FSCIL) | 2025 |

###### Dynamic network structure-based 2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
|01|[Boosting continual learning of vision-language models via mixture-of-experts adapters](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters_CVPR_2024_paper.pdf) | CVPR | DDAS| [GitHub](https://github.com/JiazuoYu/MoE-Adapters4CL) | 2024 |
|02| [APLe: Token-Wise Adaptive for Multi-Modal Prompt Learning](https://arxiv.org/pdf/2401.06827) | arXiv  | APLe| -  | 2024 |
|03| [Revisiting prompt pretraining of vision-language models](https://arxiv.org/pdf/2409.06166) | arXiv | RPP  | -  | 2024 |

###### Dynamic network structure-based 2023
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 |  [On the Soft-Subnetwork for Few-Shot Class Incremental Learning](https://arxiv.org/pdf/2209.07529) | arXiv | SoftNet | [GitHub](https://github.com/ihaeyong/SoftNet-FSCIL) | 2023 |
| 02  |[Continual Learning via Winning Subnetworks That Arise Through Stochastic Local Competition](https://openreview.net/pdf?id=fUwfjPzI8g) | ICLR | TWTA-CIL      | -  | 2023 |
| 03  | [Domain incremental lifelong learning in an open world](https://arxiv.org/pdf/2305.06555) | arXiv | Diana | [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/diana) | 2023 |
| 04  |[Prompts can play lottery tickets well: Achieving lifelong information extraction via lottery prompt tuning](https://aclanthology.org/2023.acl-long.16.pdf) | ACL | LPT| [GitHub](https://github.com/jokieleung/Lottery_Prompt)  | 2023 |
| 05  | [Orthogonal subspace learning for language model continual learning](https://arxiv.org/pdf/2310.14152) | arXiv | O-LoRA | [GitHub](https://github.com/cmnfriend/O-LoRA)| 2023 |
| 06  |[Continual diffusion: Continual customization of text-to-image diffusion with c-lora](https://arxiv.org/pdf/2304.06027) | arXiv | C-LoRA| [GitHub](https://jamessealesmith.github.io/continual-diffusion/)  | 2023 |
| 07  | [Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) | CVPR  | CODA-Prompt| [GitHub](https://github.com/GT-RIPL/CODA-Prompt)   | 2023 |


###### Dynamic network structure-based 2022 and earlier
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | ardywibowo2022varigrow| [Varigrow: Variational architecture growing for task-agnostic continual learning based on bayesian novelty](https://proceedings.mlr.press/v162/ardywibowo22a/ardywibowo22a.pdf) | ICML  | VariGrow | -  | 2022 |
| 02  | zhu2022continual   | [Continual prompt tuning for dialog state tracking](https://arxiv.org/pdf/2203.06654) | arXiv | DST | [GitHub](https://github.com/thu-coai/CPT4DST)  | 2022 |
| 03  | wang2022dualprompt | [Dualprompt: Complementary prompting for rehearsal-free continual learning](https://link.springer.com/chapter/10.1007/978-3-031-19809-0_36) | ECCV | DualPrompt  | [GitHub](https://github.com/google-research/l2p)  | 2022 |
| 04  | yan2021dynamically | [Der: Dynamically expandable representation for class incremental learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_DER_Dynamically_Expandable_Representation_for_Class_Incremental_Learning_CVPR_2021_paper.pdf) | CVPR         | DER  | [GitHub](https://github.com/Rhyssiyan/DER-ClassIL.pytorch) | 2021 |
| 05  | madotto2020continual| [Continual learning in task-oriented dialogue systems](https://arxiv.org/pdf/2012.15504) | arXiv | ToDs   | -  | 2020 |


### 2.2 Parameter-Space-Based Approaches 
#### 2.2.1 Feature Space
###### Feature Space 2025
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
|01|[Contrastive augmented graph2graph memory interaction for few shot continual learning](https://ieeexplore.ieee.org/abstract/document/10841449/) | TCSVT | LGP | - | 2025 |

###### Feature Space 2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | zhou2024expandable | [Expandable subspace ensemble for pre-trained model-based class-incremental learning](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Expandable_Subspace_Ensemble_for_Pre-Trained_Model-Based_Class-Incremental_Learning_CVPR_2024_paper.pdf) | CVPR | EASE | [GitHub](https://github.com/sun-hailong/CVPR24-Ease) | 2024 |
| 02 | zhang2024rethinking | [Rethinking misalignment in vision-language model adaptation from a causal perspective](https://proceedings.neurips.cc/paper_files/paper/2024/file/453a27b717972ef94a9a9113d236ad2f-Paper-Conference.pdf) | NeurIPS | CDC | [NeurIPS Code Policy](https://nips.cc/public/guides/CodeSubmissionPolicy) | 2024 |
| 03 | zhang2024dept | [DePT: Decoupled prompt tuning](http://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_DePT_Decoupled_Prompt_Tuning_CVPR_2024_paper.pdf) | CVPR | DePT | [GitHub](https://github.com/Koorye/DePT) | 2024 |
| 04 | zhou2024delve | [Delve into base-novel confusion: Redundancy exploration for few-shot class-incremental learning](https://arxiv.org/pdf/2405.04918) | arXiv | RDI | - | 2024 |
| 05 | shi2024deep | [Deep Correlated Prompting for Visual Recognition with Missing Modalities](https://proceedings.neurips.cc/paper_files/paper/2024/file/7ca55c8276acf1f0aa996cd3622d1df4-Paper-Conference.pdf) | NeurIPS | DCP | [GitHub](https://github.com/hulianyuyy/Deep_Correlated_Prompting) | 2024 |
| 06 | sun2024craft | [Craft: Cross-modal Aligned Features Improve Robustness of Prompt Tuning](https://arxiv.org/pdf/2407.15894) | arXiv | CRAFT | [GitHub](https://github.com/Jingchensun/Craft) | 2024 |
| 07 | lin2024m2sd | [M2sd: Multiple mixing self-distillation for few-shot class-incremental learning](https://ojs.aaai.org/index.php/AAAI/article/view/28129) | AAAI | M2SD | - | 2024 |


###### Feature Space 2023
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | mcdonnell2023ranpac | [Ranpac: Random projections and pre-trained models for continual learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/2793dc35e14003dd367684d93d236847-Paper-Conference.pdf) | NeurIPS | RanPAC | [GitHub](https://github.com/RanPAC/RanPAC) | 2023 |
| 02 | gu2023big | [Big-model Driven Few-shot Continual Learning](https://arxiv.org/pdf/2309.00862) | arXiv | B-FSCL | - | 2023 |

###### Feature Space 2022
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | qin2022continual | [Continual few-shot relation learning via embedding space regularization and data augmentation](https://arxiv.org/pdf/2203.02135) | arXiv | CFRL | [GitHub](https://github.com/qcwthu/Continual_Fewshot_Relation_Learning) | 2022 |
| 02 | wang2022learning | [Learning to decompose visual features with latent textual prompts](https://arxiv.org/pdf/2210.04287) | arXiv | DeFo | - | 2022 |
| 03 | hersche2022constrained | [Constrained few-shot class-incremental learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Hersche_Constrained_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf) | CVPR | C-FSCIL | [GitHub](https://github.com/IBM/constrained-FSCIL) | 2022 |
| 04 | ahmad2022few | [Few-shot class incremental learning leveraging self-supervised features](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Ahmad_Few-Shot_Class_Incremental_Learning_Leveraging_Self-Supervised_Features_CVPRW_2022_paper.pdf) | CVPR | FeSSSS | [GitHub](https://github.com/TouqeerAhmad/FeSSSS) | 2022 |

###### Feature Space 2021 and earlier
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | zhu2021self | [Self-promoted prototype refinement for few-shot class-incremental learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Self-Promoted_Prototype_Refinement_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf) | CVPR | SPPR | - | 2021 |
| 02 | caccia2020online | [Online fast adaptation and knowledge accumulation (osaka): a new approach to continual learning](https://proceedings.neurips.cc/paper_files/paper/2020/file/c0a271bc0ecb776a094786474322cb82-Paper.pdf) | NeurIPS | OSAKA | [GitHub](https://github.com/ElementAI/osaka) | 2020 |


#### 2.2.2 Weight Space
###### Weight Space 2025
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 |[Complementary Subspace Low-Rank Adaptation of Vision-Language Models for Few-Shot Classification](https://arxiv.org/pdf/2501.15040) | arXiv | Comp-LoRA | - | 2025 |
| 02 |[Efficient Few-Shot Continual Learning in Vision-Language Models](https://arxiv.org/pdf/2502.04098) | arXiv | LoRSU | - | 2025 |
| 03 |[Continuous Knowledge-Preserving Decomposition for Few-Shot Continual Learning](https://arxiv.org/pdf/2501.05017) | arXiv | CKPD-FSCIL | [GitHub](https://github.com/xiaojieli0903/CKPD-FSCIL) | 2025 |
| 04 |[Singular Value Fine-tuning for Few-Shot Class-Incremental Learning](https://arxiv.org/pdf/2503.10214) | arXiv | SVFCL | - | 2025 |

###### Weight Space 2024 and earlier
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 02 |[LW2G: Learning Whether to Grow for Prompt-based Continual Learning](https://openreview.net/pdf?id=QZuZmfLLRG) | ICLR | LW2G | [GitHub](https://github.com/thu-ml/HiDe-Prompt) | 2024 |
| 03 |[Continual Few-shot Relation Extraction via Adaptive Gradient Correction and Knowledge Decomposition](https://aclanthology.org/2024.findings-acl.702.pdf) | ACL | ROD_AOD | - | 2024 |
| 01 |[Warping the space: Weight space rotation for class-incremental few-shot learning](https://openreview.net/pdf?id=kPLzOfPfA2l) | ICLR | WaRP | [GitHub](https://github.com/EdwinKim3069/WaRP-CIFSL) | 2023 |


#### 2.2.3 Knowledge Distillation
###### Knowledge Distillation 2025
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Fully Fine-Tuning Beats Parameter Efficient Fine-Tuning for CLIP in Data-Limited Scenarios](https://openreview.net/pdf?id=VbszSB4pK6) | ICLR | CLIP-CITE      | - | 2025 |

###### Knowledge Distillation 2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Improving Continual Few-shot Relation Extraction through Relational Knowledge Distillation and Prototype Augmentation](https://aclanthology.org/2024.lrec-main.767.pdf) | LREC-COLING  | RK2DA          | -                                                                 | 2024 |
| 02  |[On Distilling the Displacement Knowledge for Few-Shot Class-Incremental Learning](https://arxiv.org/pdf/2412.11017) | arXiv| DKD  | - | 2024 |
| 03  |[M2sd: Multiple mixing self-distillation for few-shot class-incremental learning](https://ojs.aaai.org/index.php/AAAI/article/view/28129) | AAAI         | M2SD | -  | 2024 |
| 04  |[Revisiting prompt pretraining of vision-language models](https://arxiv.org/pdf/2409.06166) | arXiv | RPP   | -  | 2024 |
| 05  |[Improving zero-shot generalization of learned prompts via unsupervised knowledge distillation](https://link.springer.com/chapter/10.1007/978-3-031-72907-2_27) |ECCV| KDPL| [GitHub](https://github.com/miccunifi/KDPL) | 2024 |
| 06  |[Cascade prompt learning for vision-language model adaptation](https://arxiv.org/pdf/2409.17805) | ECCV  | CasPL  | [GitHub](https://github.com/megvii-research/CasPL) | 2024 |

###### Knowledge Distillation 2023 and earlier
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Serial contrastive knowledge distillation for continual few-shot relation extraction](https://arxiv.org/pdf/2305.06616) | arXiv  | SCKD  | [GitHub](https://github.com/nju-websoft/SCKD) | 2023 |
| 02  | [Feature distribution distillation-based few shot class incremental learning](https://ieeexplore.ieee.org/abstract/document/9904282) | PRAI| - | -| 2022 |
| 03  |[Knowledge Distillation: A Survey](https://arxiv.org/pdf/2006.05525)   | arXiv | Survey | - | 2021 |



## 2. Optimization Based Approaches  
### 2.1 Gradient-Based Approaches  
#### 2.1.1 Function Regularization 
###### Function Regularization 2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[HPT++: Hierarchically Prompting Vision-Language Models with Multi-Granularity Knowledge Generation and Improved Structure Modeling](https://arxiv.org/pdf/2408.14812) | arXiv | HPT++  | - | 2024 |
| 02  | [RESTORE: Towards Feature Shift for Vision-Language Prompt Learning](https://arxiv.org/pdf/2403.06136) | arXiv | RESTORE | [GitHub](https://github.com/Yaphabates/RESTORE_) | 2024 |
| 03  | [Conceptual Codebook Learning for Vision-Language Models](https://link.springer.com/chapter/10.1007/978-3-031-72980-5_14) | ECCV | CoCoLe| -  | 2024 |
| 04  | wang2024understanding| [Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models](https://arxiv.org/pdf/2410.02681) | arXiv | DOR| [GitHub](https://github.com/ml-stat-Sustech/Outlier-Calibration)  | 2024 |
| 05  | chen2024conditional| [Conditional prototype rectification prompt learning](https://ieeexplore.ieee.org/abstract/document/11069290/) | TCSVT | CPR | [GitHub](https://github.com/chenhaoxing/CPR) | 2024 |
| 06  | kuchibhotlasemantic| [Semantic Alignment for Prompt-Tuning in Vision Language Models](https://openreview.net/pdf?id=avDr56QjSI) | TMLR | SAP| - | 2024 |
###### Function Regularization 2023
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Self-regulating prompts: Foundational model adaptation without forgetting](https://openaccess.thecvf.com/content/ICCV2023/papers/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.pdf) | ICCV | PromptSR  | [GitHub](https://github.com/muzairkhattak/PromptSRC)| 2023 |
| 02  |[Towards Efficient Vision-Language Tuning: More Information Density, More Generalizability](https://arxiv.org/pdf/2312.10813) | arXiv| DIP| -| 2023 |


#### 2.1.2 Weight Regularization 
###### Weight Regularization 2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [M³PL: Identifying and Exploiting View Bias of Prompt Learning](https://openreview.net/pdf?id=2rnTIBm19V) | Transact Mach Learn Res | M3PL | -  | 2024 |
| 02  | [Fine-Tuning CLIP's Last Visual Projector: A Few-Shot Cornucopia](https://inria.hal.science/hal-04986168/document) | Inria | -  | - | 2024 |
| 03  | [On Distilling the Displacement Knowledge for Few-Shot Class-Incremental Learning](https://arxiv.org/pdf/2412.11017) | arXiv | DKD   | -| 2024 |

###### Weight Regularization 2023
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) | CVPR  | CODA-Prompt    | [GitHub](https://github.com/GT-RIPL/CODA-Prompt) | 2023 |
| 02  |[Few shot class incremental learning via efficient prototype replay and calibration](https://www.mdpi.com/1099-4300/25/5/776) | Entropy| EPRC  | -  | 2023 |
| 03  |[Continual diffusion: Continual customization of text-to-image diffusion with c-lora](https://arxiv.org/pdf/2304.06027) | arXiv  | C-LoRA| [GitHub](https://jamessealesmith.github.io/continual-diffusion/)  | 2023 |

#### 2.1.3 Gradient space  
###### Gradient space  2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Generalizable Prompt Learning via Gradient Constrained Sharpness-aware Minimization](https://ieeexplore.ieee.org/abstract/document/10814656/) | TMM          | GCSCoOp | [GitHub](https://github.com/llcllc1997/GCSCoOp) | 2024 |
| 02 | [LW2G: Learning Whether to Grow for Prompt-based Continual Learning](https://openreview.net/pdf?id=QZuZmfLLRG) | ICLR | LW2G | [GitHub](https://github.com/thu-ml/HiDe-Prompt)  | 2024 |
| 03| [Continual Few-shot Relation Extraction via Adaptive Gradient Correction and Knowledge Decomposition](https://aclanthology.org/2024.findings-acl.702.pdf) | ACL | ROD&AOD | - | 2024 |
| 04  | [Fine-Tuning CLIP's Last Visual Projector: A Few-Shot Cornucopia](https://inria.hal.science/hal-04986168/document) | Inria   | -  | -  | 2024 |

###### Gradient space  2023
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Prompt gradient projection for continual learning](https://openreview.net/pdf?id=EH2O3h7sBI) | ICLR | PGP | [GitHub](https://github.com/JingyangQiao/prompt-gradient-projection) | 2023 |
| 02  |[Prompt-aligned gradient for prompt tuning](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Prompt-aligned_Gradient_for_Prompt_Tuning_ICCV_2023_paper.pdf) | ICCV | ProGrad  | - | 2023 |
| 03  |[Visual-language prompt tuning with knowledge-guided context optimization](https://openaccess.thecvf.com/content/CVPR2023/papers/Yao_Visual-Language_Prompt_Tuning_With_Knowledge-Guided_Context_Optimization_CVPR_2023_paper.pdf) | CVPR | KgCoOp  | -  | 2023 |

###### Gradient space  2021
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Gradient projection memory for continual learning](https://arxiv.org/pdf/2103.09762) | arXiv| GPM | [GitHub](https://github.com/sahagobinda/GPM) | 2021 |
| 02  | tang2021layerwise  | [Layerwise optimization by gradient decomposition for continual learning](http://openaccess.thecvf.com/content/CVPR2021/papers/Tang_Layerwise_Optimization_by_Gradient_Decomposition_for_Continual_Learning_CVPR_2021_paper.pdf) | CVPR | -  | -  | 2021 |

#### 2.1.4 Loss Function
###### Loss Function  2025
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Fully Fine-Tuning Beats Parameter Efficient Fine-Tuning for CLIP in Data-Limited Scenarios](https://openreview.net/pdf?id=VbszSB4pK6) | ICLR | CLIP-CITE  | - | 2025 |
| 02  |[Style-Pro: Style-Guided Prompt Learning for Generalizable Vision-Language Models](https://ieeexplore.ieee.org/abstract/document/10943992/) | WACV  | Style-Pro | - | 2025 |

###### Loss Function  2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Flatten long-range loss landscapes for cross-domain few-shot learning](https://openaccess.thecvf.com/content/CVPR2024/papers/Zou_Flatten_Long-Range_Loss_Landscapes_for_Cross-Domain_Few-Shot_Learning_CVPR_2024_paper.pdf) | CVPR  | FLoR | [GitHub](https://github.com/Zoilsen/FLoR) | 2024 |
| 02  |[A bag of tricks for few-shot class-incremental learning](https://arxiv.org/pdf/2403.14392) | arXiv | -  | - | 2024 |
| 03  |[Craft: Cross-modal Aligned Features Improve Robustness of Prompt Tuning](https://arxiv.org/pdf/2407.15894) | arXiv| CRAFT | [GitHub](https://github.com/Jingchensun/Craft) | 2024 |
| 04  |[Pre-trained vision and language transformers are few-shot incremental learners](https://openaccess.thecvf.com/content/CVPR2024/papers/Park_Pre-trained_Vision_and_Language_Transformers_Are_Few-Shot_Incremental_Learners_CVPR_2024_paper.pdf) | CVPR | PriViLege| [GitHub](https://github.com/KHU-AGI/PriViLege) | 2024 |
| 05  |[One-stage prompt-based continual learning](https://link.springer.com/chapter/10.1007/978-3-031-72624-8_10) | ECCV | PCL  | - | 2024 |
| 06  | [Conceptual Codebook Learning for Vision-Language Models](https://link.springer.com/chapter/10.1007/978-3-031-72980-5_14) | ECCV  | CoCoLe | - | 2024 |

###### Loss Function  2023
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 04  | [Few-shot class-incremental learning for medical time series classification](https://drive.google.com/file/d/1hsaJUvUPMAcMHuAoqL0wKUssG_o9rR71/view) | J-BHI | MAPIC | - | 2023 |
| 05  | [Ica-proto: Iterative cross alignment prototypical network for incremental few-shot relation classification](https://aclanthology.org/2023.findings-eacl.171.pdf) | EACL | ICA-Proto| -  | 2023 |
| 06  |[Consistent prototype learning for few-shot continual relation extraction](https://aclanthology.org/2023.acl-long.409.pdf) | ACL| ConPL | [GitHub](https://github.com/XiudiChen/ConPL)| 2023 |


###### Loss Function  2021
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Memory-efficient incremental learning through feature adaptation](https://link.springer.com/chapter/10.1007/978-3-030-58517-4_41) | ECCV |Feature_Adaptation | -  | 2020 |
| 02  |[Overcoming catastrophic forgetting in incremental few-shot learning by finding flat minima](https://proceedings.neurips.cc/paper_files/paper/2021/file/357cfba15668cc2e1e73111e09d54383-Paper.pdf) | NeurIPS  | F2M  | [GitHub](https://github.com/moukamisama/F2M) | 2021 |
| 03  |[Der: Dynamically expandable representation for class incremental learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_DER_Dynamically_Expandable_Representation_for_Class_Incremental_Learning_CVPR_2021_paper.pdf) | CVPR | DER | [GitHub](https://github.com/Rhyssiyan/DER-ClassIL.pytorch) | 2021 |

### 2.2 Tuning-based Approaches 
#### 2.2.1 Adapter-Based
###### Adapter-Based 2025
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[CMoA: Contrastive Mixture of Adapters for Generalized Few-Shot Continual Learning](https://ieeexplore.ieee.org/abstract/document/10891550/) | IEEE Transactions on Multimedia | CMoA | - | 2025 |
| 02  |[Continuous Knowledge-Preserving Decomposition for Few-Shot Continual Learning](https://arxiv.org/pdf/2501.05017) | arXiv| CKPD-FSCIL| [GitHub](https://github.com/xiaojieli0903/CKPD-FSCIL)| 2025 |
| 03  |[Complementary Subspace Low-Rank Adaptation of Vision-Language Models for Few-Shot Classification](https://arxiv.org/pdf/2501.15040) | arXiv| Comp-LoRA| -  | 2025 |
| 04  |[Singular Value Fine-tuning for Few-Shot Class-Incremental Learning](https://arxiv.org/pdf/2503.10214) | arXiv | SVFCL | - | 2025 |

###### Adapter-Based 2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[RESTORE: Towards Feature Shift for Vision-Language Prompt Learning](https://arxiv.org/pdf/2403.06136) | arXiv | RESTORE| [GitHub](https://github.com/Yaphabates/RESTORE_)  | 2024 |
| 02  |[Conditional prototype rectification prompt learning](https://ieeexplore.ieee.org/abstract/document/11069290/) | TCSVT| CPR| [GitHub](https://github.com/chenhaoxing/CPR) | 2024 |
| 03  |[Boosting continual learning of vision-language models via mixture-of-experts adapters](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters_CVPR_2024_paper.pdf) | CVPR | DDAS | [GitHub](https://github.com/JiazuoYu/MoE-Adapters4CL) | 2024 |
| 04  |[Coin: A benchmark of continual instruction tuning for multimodel large language models](https://proceedings.neurips.cc/paper_files/paper/2024/file/6a45500d9eda640deed90d8a62742be5-Paper-Datasets_and_Benchmarks_Track.pdf) | NeurIPS | CoIN| [GitHub](https://github.com/zackschen/CoIN)  | 2024 |

###### Adapter-Based 2023 
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Continual diffusion: Continual customization of text-to-image diffusion with c-lora](https://arxiv.org/pdf/2304.06027) | arXiv| C-LoRA| [GitHub](https://jamessealesmith.github.io/continual-diffusion/)  | 2023 |
| 02  |[Towards Difficulty-Agnostic Efficient Transfer Learning for Vision-Language Models](https://arxiv.org/pdf/2311.15569) | arXiv| APEX  | - | 2023 |


###### Adapter-Based 2022 and earlier
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Lora: Low-rank adaptation of large language models](https://arxiv.org/pdf/2106.09685v1) | ICLR | LoRA| [GitHub](https://github.com/microsoft/LoRA)| 2022 |
| 02  |[Towards a unified view of parameter-efficient transfer learning](https://arxiv.org/pdf/2110.04366) | arXiv | - | [GitHub](https://github.com/jxhe/unify-parameter-efficient-tuning) | 2021 |


#### 2.2.2 Prompt-Based
###### Prompt-Based 2025
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Fully Fine-Tuning Beats Parameter Efficient Fine-Tuning for CLIP in Data-Limited Scenarios](https://openreview.net/pdf?id=VbszSB4pK6) | ICLR | CLIP-CITE| - | 2025 |

###### Prompt-Based 2024
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Prompt learning via meta-regularization](https://openaccess.thecvf.com/content/CVPR2024/papers/Park_Prompt_Learning_via_Meta-Regularization_CVPR_2024_paper.pdf) | CVPR| ProMetaR| [GitHub](https://github.com/mlvlab/ProMetaR)| 2024 |
| 02  |[Modal-aware prompt tuning with deep adaptive feature enhancement](https://www.sciencedirect.com/science/article/abs/pii/S0045790624001988) | COMPUT ELECTR ENG | MAP-DAFE | - | 2024 |
| 03 | [APLe: Token-Wise Adaptive for Multi-Modal Prompt Learning](https://arxiv.org/pdf/2401.06827) | arXiv| APLe| - | 2024 |
| 04 | [Concept-guided prompt learning for generalization in vision-language models](https://ojs.aaai.org/index.php/AAAI/article/download/28568/29104) | AAAI | CPL| [GitHub](https://github.com/rambo-coder/CPL)| 2024 |
| 05  |[Learning prompt with distribution-based feature replay for few-shot class-incremental learning](https://arxiv.org/pdf/2401.01598) | arXiv| LP-DiF| [GitHub](https://github.com/1170300714/LP-DiF)| 2024 |
| 06  |[SEP: Self-Enhanced Prompt Tuning for Visual-Language Model](https://arxiv.org/pdf/2405.15549) | arXiv  | SEP | [GitHub](https://github.com/htyao89/SEP/) | 2024 |
| 07  |[Craft: Cross-modal Aligned Features Improve Robustness of Prompt Tuning](https://arxiv.org/pdf/2407.15894) | arXiv| CRAFT | [GitHub](https://github.com/Jingchensun/Craft)| 2024 |
| 08  | [Improving zero-shot generalization of learned prompts via unsupervised knowledge distillation](https://link.springer.com/chapter/10.1007/978-3-031-72907-2_27) | ECCV | KDPL | [GitHub](https://github.com/miccunifi/KDPL)| 2024 |
| 09  |[Pre-trained vision and language transformers are few-shot incremental learners](https://openaccess.thecvf.com/content/CVPR2024/papers/Park_Pre-trained_Vision_and_Language_Transformers_Are_Few-Shot_Incremental_Learners_CVPR_2024_paper.pdf) | CVPR | PriViLege| [GitHub](https://github.com/KHU-AGI/PriViLege)| 2024 |
| 10  |[Cascade prompt learning for vision-language model adaptation](https://arxiv.org/pdf/2409.17805) | ECCV | CasPL| [GitHub](https://github.com/megvii-research/CasPL)| 2024 |
| 11 | [Rethinking misalignment in vision-language model adaptation from a causal perspective](https://proceedings.neurips.cc/paper_files/paper/2024/file/453a27b717972ef94a9a9113d236ad2f-Paper-Conference.pdf) | NeurIPS| CDC | [GitHub](https://nips.cc/public/guides/CodeSubmissionPolicy)  2024|
| 12 |[TCP: Textual-based Class-aware Prompt tuning for Visual-Language Model](https://openaccess.thecvf.com/content/CVPR2024/papers/Yao_TCPTextual-based_Class-aware_Prompt_tuning_for_Visual-Language_Model_CVPR_2024_paper.pdf) | CVPR | TCP| [GitHub](https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning) | 2024 |

###### Prompt-Based 2023
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Visual instruction tuning](https://proceedings.neurips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf) | NeurIPS          | LLaVA | [GitHub](https://llava-vl.github.io/) | 2023 |
| 02  | [Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) | CVPR | CODA-Prompt| [GitHub](https://github.com/GT-RIPL/CODA-Prompt) | 2023 |
| 03  |[MaPLe: Multi-modal prompt learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Khattak_MaPLe_Multi-Modal_Prompt_Learning_CVPR_2023_paper.pdf) | CVPR| MaPLe| [GitHub](https://github.com/muzairkhattak/multimodal-prompt-learning) | 2023 |
| 04  |[Self-regulating prompts: Foundational model adaptation without forgetting](https://openaccess.thecvf.com/content/ICCV2023/papers/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.pdf) | ICCV | PromptSR| [GitHub](https://github.com/muzairkhattak/PromptSRC) | 2023 |
| 05  | [Consistency-guided prompt learning for vision-language models](https://arxiv.org/pdf/2306.01195) | arXiv   | CoPrompt | [GitHub](https://github.com/ShuvenduRoy/CoPrompt) | 2023 |
| 06  |[Multimodal parameter-efficient few-shot class incremental learning](https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/DAlessandro_Multimodal_Parameter-Efficient_Few-Shot_Class_Incremental_Learning_ICCVW_2023_paper.pdf) | ICCV | CPE-CLIP | - | 2023 |

###### Prompt-Based 2022
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Learning to prompt for continual learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.pdf) | CVPR | L2P| [GitHub](https://github.com/google-research/l2p) | 2022 |
| 02  |[Dualprompt: Complementary prompting for rehearsal-free continual learning](https://link.springer.com/chapter/10.1007/978-3-031-19809-0_36) | ECCV| DualPrompt| [GitHub](https://github.com/google-research/l2p)| 2022 |
| 03| [Learning to prompt for vision-language models](https://arxiv.org/pdf/2109.01134) | IJCV| CoOp | [GitHub](https://github.com/KaiyangZhou/CoOp) | 2022 |
| 04  | [Visual prompt tuning](https://arxiv.org/pdf/2203.12119) | arXiv| VPT  | [GitHub](https://github.com/kmnp/vpt)  | 2022 |

###### Prompt-Based 2021
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Finetuned language models are zero-shot learners](https://arxiv.org/pdf/2109.01652) | arXiv | FLAN  | [GitHub](https://github.com/google-research/flan) | 2021 |
| 02  |[The power of scale for parameter-efficient prompt tuning](https://arxiv.org/pdf/2104.08691) | arXiv | - | [GitHub](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L264) | 2021 |
| 03  |[Bloom: A 176b-parameter open-access multilingual language model](https://arxiv.org/pdf/2211.05100) | arXiv | BLOOM| -| 2021 |

#### 2.2.3 Instruct Tuning
###### Instruct Tuning 2025
| No. | Title |Type | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|


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


