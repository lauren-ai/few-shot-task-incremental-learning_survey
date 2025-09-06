# Few-shot Task-incremental Learning: Methods, Challenges, and Future Directions
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)
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
          - [Feature Replay 2024](#feature-replay-2024)
          - [Feature Replay 2023](#feature-replay-2023)
          - [Feature Replay 2022](#feature-replay-2022)
          - [Feature Replay 2021](#feature-replay-2021)
      - [1.1.3 Generative Replay](#113-generative-replay)
          - [Generative Replay 2025](#generative-replay-2025)
          - [Generative Replay 2024](#generative-replay-2024)
          - [Generative Replay 2023](#generative-replay-2023)
          - [Generative Replay 2022](#generative-replay-2022)
          - [Generative Replay 2021](#generative-replay-2021)
      - [1.1.4 Pseudo-scenarios Replay](#114-pseudo-scenarios-reply)
          - [Pseudo-scenarios Replay 2025](#pseudo-scenarios-reply-2025)
          - [Pseudo-scenarios Replay 2024](#pseudo-scenarios-reply-2024)
          - [Pseudo-scenarios Replay 2023](#pseudo-scenarios-reply-2023)
          - [Pseudo-scenarios Replay 2022](#pseudo-scenarios-reply-2022)
          - [Pseudo-scenarios Replay 2021](#pseudo-scenarios-reply-2022)
          - [Pseudo-scenarios Replay 2021 and earlier](#pseudo-scenarios-reply-2021-and-earlier)
      - [1.1.5 Raw-data Replay](#115-raw-data-replay)
          - [Raw-data Replay 2025](#raw-data-replay-2025)
          - [Raw-data Replay 2024](#raw-data-replay-2024)
          - [Raw-data Replay 2023](#raw-data-replay-2023)
          - [Raw-data Replay 2022](#raw-data-replay-2022)
          - [Raw-data Replay 2021](#raw-data-replay-2021)
    - [1.2 Data-Augmentation-Based Approaches](#12-data-augmentation-based-approaches)
      - [1.2.1 Data Augmentation](#121-data-augmentation)
          - [Data Augmentation 2024](#data-augmentation-2024)
          - [Data Augmentation 2023](#data-augmentation-2023)
          - [Data Augmentation 2022 and earlier](#data-augmentation-2022-and-earlier)
      - [1.2.2 Feature Augmentation](#122-feature-augmentation)
          - [Feature Augmentation 2025](#feature-augmentation-2025)
          - [Feature Augmentation 2024](#feature-augmentation-2024)
          - [Feature Augmentation 2023](#feature-augmentation-2023)
          - [Feature Augmentation 2022 and earlier](#feature-augmentation-2022-and-earlier)
  - [2. model Based Approaches](#2-model-based-approaches)
    - [2.1 Architecture-Based Approches](#21-architecture-based-approaches)
      - [2.1.1 Attention-Based](#211-attention-based)
        - [Attention-Based 2024](#attention-based-2024)
        - [Attention-Based 2023 and earlier](#attention-based-2023-and-earlier)
      - [2.1.2 Dynamic network structure-based](#212-dynamic-network-structure-based)
          - [Dynamic network structure-based 2025](#dynamic-network-structure-based-2025)
          - [Dynamic network structure-based 2024](#dynamic-network-structure-based-2024)
          - [Dynamic network structure-based 2023](#dynamic-network-structure-based-2023)
          - [Dynamic network structure-based 2022 and earlier](#dynamic-network-structure-based-2022-and-earlier)
    - [2.2 Parameter-Based Approaches](#22-parameter-space-based-approaches)
        - [2.2.1 Feature Space](#221-feature-space)
           - [Feature Space 2025](#feature-space-2025)
           - [Feature Space 2024](#feature-space-2024)
           - [Feature Space 2023](#feature-space-2023)
           - [Feature Space 2022](#feature-space-2022)
           - [Feature Space 2021 and earlier](#feature-space-2021)
        - [2.2.2 Weight Space](#222-weight-space)
           - [Weight Space 2025](#weight-space-2025)
           - [Weight Space 2024 and earlier](#weight-space-2024-and-earlier)
        - [2.2.3 Knowledge Distillation](#223-knowledge-distillation)
           - [Knowledge Distillation 2025](#knowledge-distillation-2025)
           - [Knowledge Distillation 2024](#knowledge-distillation-2024)
           - [Knowledge Distillation 2023 and earlier](#knowledge-distillation-2023-and-earlier)
  - [3. Optimization Based Approaches](#3-optimization-based-approaches)
    - [3.1 Gradient-Based Approaches](#31-gradient-based-approaches)
      - [3.1.1 Function Regularization](#311-function-regularization)
           - [Function Regularization 2024](#function-regularization-2024)
           - [Function Regularization 2023](#function-regularization-2023)
      - [3.1.2 Weight Regularization](#312-weight-regularization)
           - [Weight Regularization 2024](#weight-regularization-2024)
           - [Weight Regularization 2023](#weight-regularization-2023)
      - [3.1.3 Gradient Space](#313-gradient-space)
           - [Gradient Space 2024](#gradient-space-2024)
           - [Gradient Space 2023](#gradient-space-2023)
           - [Gradient Space 2021](#gradient-space-2021)
      - [3.1.4 Loss Function](#314-loss-function)
           - [Loss Function 2025](#loss-function-2025)
           - [Loss Function 2024](#loss-function-2024)
           - [Loss Function 2023](#loss-function-2023)
           - [Loss Function 2021](#loss-function-2021)
    - [3.2 Tuning-Based Approaches](#32-tuning-based-approaches)
      - [3.2.1 Adapter-Based](#321-adapter-based)
           - [Adapter-Based 2025](#adapter-based-2025)
           - [Adapter-Based 2024](#adapter-based-2024)
           - [Adapter-Based 2023](#adapter-based-2023)
           - [Adapter-Based 2022 and earlier](#adapter-based-2022-and-earlier)
      - [3.2.2 Prompt-Based](#322-prompt-based)
           - [Prompt-Based 2025](#prompt-based-2025)
           - [Prompt-Based 2024](#prompt-based-2024)
           - [Prompt-Based 2023](#prompt-based-2023)
           - [Prompt-Based 2022](#prompt-based-2022)
           - [Prompt-Based 2021](#prompt-based-2021)
      - [3.2.3 Instruct Tuning](#323-instruct-tuning)
           - [Instruct Tuning 2024](#instruct-tuning-2024)
           - [Instruct Tuning 2023](#instruct-tuning-2023)
           - [Instruct Tuning 2022 and earlier](#instruct-tuning-2022-and-earlier)
      - [3.2.4 Prefix Tuning](#324-prefix-tuning)
           - [Prefix Tuning 2024](#prefix-tuning-2024)
           - [Prefix Tuning 2023 and earlier](#prefix-tuning-2023-and-earlier)
  - [4. Survey of Incremental learning](#4-survey-of-incremental-learning)
    - [Survey of Incremental learning 2025](#survey-of-incremental-learning-2025)
    - [Survey of Incremental learning 2024](#survey-of-incremental-learning-2024)
    - [Survey of Incremental learning 2023](#survey-of-incremental-learning-2023)
    - [Survey of Incremental learning 2022](#survey-of-incremental-learning-2022)
    - [Survey of Incremental learning 2020](#survey-of-incremental-learning-2020)
  - [5. Other Works](#5-other-works)
    - [Papers](#papers)

## 0. Overview
The repo includes the ongoing updates of representative few-shot task incremental learning papers and open-source codes.  

**Taxonomy**: In our survey, we provide a comprehensive review of the state-of-the-art in few-shot task incremental learning, which we categorize along three orthogonal axes: Data-Based Approaches, Model-Based Approaches, and Optimization-Based Approaches. 

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
| No. | Title   | Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Continual Learning of Personalized Generative Face Models with Experience Replay](https://ieeexplore.ieee.org/abstract/document/10944168) | WACV | - | [GitHub](https://anniedde.github.io/personalizedcontinuallearning.github.io/) | 2025 |
| 02 | [AnchorInv: Few-Shot Class-Incremental Learning of Physiological Signals via Feature Space-Guided Inversion](https://ojs.aaai.org/index.php/AAAI/article/view/33563/35718) | AAAI | - | [GitHub](https://github.com/chenqi-li/anchorinv) | 2025 |


###### Generative Replay 2024
| No. | Title   | Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Continual offline reinforcement learning via diffusion-based dual generative replay](https://arxiv.org/pdf/2404.10662) | arXiv | CuGRO | [GitHub](https://github.com/NJU-RL/CuGRO) | 2024 |
| 02 | [Clip with generative latent replay: a strong baseline for incremental learning](https://arxiv.org/pdf/2407.15793?) | arXiv | CGIL | [GitHub](https://github.com/aimagelab/mammoth) | 2024 |
| 03 | [Few-shot task learning through inverse generative modeling](https://proceedings.neurips.cc/paper_files/paper/2024/file/) | NeurIPS | - | - | 2024 |
| 04 | [General federated class-incremental learning with lightweight generative replay](https://ieeexplore.ieee.org/abstract/document/10612802/) | IEEE IoT-J | GenFCIL | - | 2024 |

###### Generative Replay 2023
| No. | Title | Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Task-aware information routing from common representation space in lifelong learning](https://arxiv.org/pdf/2302.11346) | arXiv | TAMiL | [GitHub](https://github.com/NeurAI-Lab/TAMiL) | 2023 |
| 02 | [Class-incremental learning using diffusion model for distillation and replay](https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/Jodelet_Class-Incremental_Learning_Using_Diffusion_Model_for_Distillation_and_Replay_ICCVW_2023_paper.pdf) | ICCV | SDDR | - | 2023 |


###### Generative Replay 2022
| No. | Title | Venue | Algorithm Name | Code |Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Few-shot class-incremental learning via entropy-regularized data-free replay](https://link.springer.com/chapter/10.1007/978-3-031-20053-3_9) | ECCV | - | - | 2022 |
| 02 | [Memory replay with data compression for continual learning](https://arxiv.org/pdf/2202.06592) | arXiv | MRDC | - | 2022 |
| 03 | [Semantics-driven generative replay for few-shot class incremental learning](https://dl.acm.org/doi/abs/10.1145/3503161.3548160) | Proc ACM Int Conf Multimed | - | - | 2022 |

###### Generative Replay 2021
| No. | Title | Venue | Algorithm Name | Code| Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Triple-memory networks: A brain-inspired method for continual learning](https://ieeexplore.ieee.org/abstract/document/9540230/) | TNNLS | TMNs | - | 2021 |
 
 
#### 1.1.4 Pseudo-scenarios Reply
###### Pseudo-scenarios Reply 2025
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Pseudo Informative Episode Construction for Few-Shot Class-Incremental Learning](https://ojs.aaai.org/index.php/AAAI/article/view/33729)  | AAAI  | PIEC  | - | 2025 |
| 02 | [SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection](https://openaccess.thecvf.com/content/CVPR2025/papers/Tran_SimLTD_Simple_Supervised_and_Semi-Supervised_Long-Tailed_Object_Detection_CVPR_2025_paper.pdf)  | CVPR   | SimLTD   | - | 2025 |

###### Pseudo-scenarios Reply 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [PASS-Net: A Pseudo Classes and Stochastic Classifiers Based Network for Few-Shot Class-Incremental Automatic Modulation Classification](https://ieeexplore.ieee.org/abstract/document/10684455/)  | TWC   | PASS-Net  | - | 2024 |
| 02 | [M2SD: Multiple Mixing Self-Distillation for Few-shot Class-Incremental Learning](https://ojs.aaai.org/index.php/AAAI/article/view/28129)  | AAAI    | M2SD    | - | 2024 |

###### Pseudo-scenarios Reply 2023
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Serial Contrastive Knowledge Distillation for Continual Few-shot Relation Extraction](https://arxiv.org/pdf/2305.06616)  | arXiv   | SCKD  | [GitHub](https://github.com/nju-websoft/SCKD) | 2023 |
| 02 | [Evolving Dictionary Representation for Few-shot Class-incremental Learning](https://arxiv.org/pdf/2305.01885)  | arXiv    | D-FSCIL    | - | 2023 |
| 03 | [Learning with Fantasy: Semantic-Aware Virtual Contrastive Constraint for Few-Shot Class-Incremental Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_Learning_With_Fantasy_Semantic-Aware_Virtual_Contrastive_Constraint_for_Few-Shot_Class-Incremental_CVPR_2023_paper.pdf)  | CVPR    | SAVC    | [GitHub](https://github.com/zysong0113/SAVC)  | 2023 |

###### Pseudo-scenarios Reply 2022
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 |  [Forward compatible few-shot class-incremental learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Forward_Compatible_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf)  | CVPR   | BGM  | - | 2022 |
| 02 | [Forward compatible few-shot class-incremental learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Forward_Compatible_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf)  | CVPR    | FACT    | [GitHub](https://github.com/zhoudw-zdw/CVPR22-Fact) | 2022 |
| 03 | [Few-shot class-incremental learning by sampling multi-phase tasks](https://ieeexplore.ieee.org/abstract/document/9864267) | TPAMI    | LIMIT    | - | 2022 |


###### Pseudo-scenarios Reply 2021 and earlier
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 |  [Few-shot incremental learning with continually evolved classifiers](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Few-Shot_Incremental_Learning_With_Continually_Evolved_Classifiers_CVPR_2021_paper.pdf)  | CVPR   | CEC  | - | 2021 |
| 02 | [LFPT5: A unified framework for lifelong few-shot language learning based on prompt tuning of T5](https://arxiv.org/pdf/2110.07298)  | arXiv    | LFPT5 | [GitHub](https://github.com/qcwthu/Lifelong-Fewshot-Language-Learning) | 2021 |
| 03 | [Self-supervised label augmentation via input transformations](https://proceedings.mlr.press/v119/lee20c/lee20c.pdf) | ICML    | SLA  | [GitHub](https://github.com/hankook/SLA) | 2020 |


#### 1.1.5 Raw-data replay
###### Raw-data replay 2025
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Continual Learning of Personalized Generative Face Models with Experience Replay](https://ieeexplore.ieee.org/abstract/document/10944168) | WACV | - | [GitHub](https://anniedde.github.io/personalizedcontinuallearning.github.io/) | 2025 |


###### Raw-data replay 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [InsCL: A data-efficient continual learning paradigm for fine-tuning large language models with instructions](https://arxiv.org/abs/2403.11435) | arXiv | InsCL | - | 2024 |
| 02 | [Learning to learn for few-shot continual active learning](https://link.springer.com/content/pdf/10.1007/s10462-024-10924-x.pdf) | Artificial Intelligence Review | Meta-CAL | [GitHub](https://pytorch.org/1.10.0+cu113) | 2024 |

###### Raw-data replay 2023
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Task-aware information routing from common representation space in lifelong learning](https://arxiv.org/pdf/2302.11346) | arXiv | TAMiL | [GitHub](https://github.com/NeurAI-Lab/TAMiL) | 2023 |

###### Raw-data replay 2022
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Memory replay with data compression for continual learning](https://arxiv.org/pdf/2202.06592) | arXiv | MRDC | - | 2022 |
| 02 | [Incremental meta-learning via episodic replay distillation for few-shot image recognition](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Wang_Incremental_Meta-Learning_via_Episodic_Replay_Distillation_for_Few-Shot_Image_Recognition_CVPRW_2022_paper.pdf) | CVPR | ERD | - | 2022 |
| 03 | [Fine-tuned language models are continual learners](https://arxiv.org/pdf/2205.12393) | arXiv | CT0 | [GitHub](https://github.com/ThomasScialom/T0_continual_learning) | 2022 |


###### Raw-data replay 2021
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Generalized and incremental few-shot learning by explicit learning and calibration without forgetting](https://openaccess.thecvf.com/content/ICCV2021/papers/Kukleva_Generalized_and_Incremental_Few-Shot_Learning_by_Explicit_Learning_and_Calibration_ICCV_2021_paper.pdf) | ICCV | LCwoF | [GitHub](https://github.com/annusha/LCwoF) | 2021 |
| 02 |  [Few-shot incremental learning with continually evolved classifiers](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Few-Shot_Incremental_Learning_With_Continually_Evolved_Classifiers_CVPR_2021_paper.pdf)  | CVPR   | CEC  | - | 2021 |

### 1.2 Data-Augmentation-Based Approaches 
#### 1.2.1 Data Augmentation
###### Data Augmentation 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
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
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Serial contrastive knowledge distillation for continual few-shot relation extraction](https://arxiv.org/pdf/2305.06616) | arXiv | SCKD | [GitHub](https://github.com/nju-websoft/SCKD) | 2023 |
| 02 | [Task-oriented multi-modal mutual leaning for vision-language models](https://openaccess.thecvf.com/content/ICCV2023/papers/Long_Task-Oriented_Multi-Modal_Mutual_Leaning_for_Vision-Language_Models_ICCV_2023_paper.pdf) | ICCV | CTP | - | 2023 |
| 03 |[LoGoPrompt: Synthetic text images can be good visual prompts for vision-language models](https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_LoGoPrompt_Synthetic_Text_Images_Can_Be_Good_Visual_Prompts_for_ICCV_2023_paper.pdf) | ICCV | LoGoPrompt | [GitHub](https://chengshiest.github.io/logo) | 2023 |
| 04 |[MaPLe: Multi-modal prompt learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Khattak_MaPLe_Multi-Modal_Prompt_Learning_CVPR_2023_paper.pdf) | CVPR | MaPLe | [GitHub](https://github.com/muzairkhattak/multimodal-prompt-learning) | 2023 |
| 05 |[Ranpac: Random projections and pre-trained models for continual learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/2793dc35e14003dd367684d93d236847-Paper-Conference.pdf) | NeurIPS | RanPAC | [GitHub](https://github.com/RanPAC/RanPAC) | 2023 |
| 06 |[Few shot class incremental learning via efficient prototype replay and calibration](https://www.mdpi.com/1099-4300/25/5/776) | Entropy | EPRC | - | 2023 |

###### Data Augmentation 2022 and earlier
| No. | Title |Venue | Algorithm Name | Code | Year |
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
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
|01| [Contrastive augmented graph2graph memory interaction for few shot continual learning](https://ieeexplore.ieee.org/abstract/document/10841449/ ) | TCSVT | LGP | -  | 2025 |


###### Feature Augmentation 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
|01| [Improving Continual Few-shot Relation Extraction through Relational Knowledge Distillation and Prototype Augmentation](https://aclanthology.org/2024.lrec-main.767.pdf) | LREC-COLING  | RK2DA | - | 2024 |
|02| [Making pre-trained language models better continual few-shot relation extractors](https://arxiv.org/pdf/2402.15713) | arXiv| CPL | [GitHub](https://github.com/mashengkun/CPL)| 2024 |
|03| [Conditional prototype rectification prompt learning](https://ieeexplore.ieee.org/abstract/document/11069290/) | TCSVT  | CPR  | [GitHub](https://github.com/chenhaoxing/CPR)  | 2024 |
|04| [Modal-aware prompt tuning with deep adaptive feature enhancement](https://www.sciencedirect.com/science/article/abs/pii/S0045790624001988)| COMPUT_ELECTR_ENG | MAP-DAFE| -| 2024 |
|05| [A hard-to-beat baseline for training-free clip-based adaptation](https://arxiv.org/pdf/2402.04087) | arXiv | GDA  | [GitHub](https://github.com/mrflogs/ICLR24) | 2024 |
|06| [APLe: Token-Wise Adaptive for Multi-Modal Prompt Learning] (https://arxiv.org/pdf/2401.06827) | arXiv | APLe | - | 2024 |


###### Feature Augmentation 2023
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Consistent prototype learning for few-shot continual relation extraction](https://aclanthology.org/2023.acl-long.409.pdf ) | ACL | ConPL  | [GitHub](https://github.com/XiudiChen/ConPL) | 2023 |
| 02  | [Ica-proto: Iterative cross alignment prototypical network for incremental few-shot relation classification](https://aclanthology.org/2023.findings-eacl.171.pdf) | EACL | ICA-Proto  | -  | 2023 |
| 03  | [Task-oriented multi-modal mutual leaning for vision-language models](https://openaccess.thecvf.com/content/ICCV2023/papers/Long_Task-Oriented_Multi-Modal_Mutual_Leaning_for_Vision-Language_Models_ICCV_2023_paper.pdf)    | ICCV | CTP  | -  | 2023 |
| 04  | [Ranpac: Random projections and pre-trained models for continual learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/2793dc35e14003dd367684d93d236847-Paper-Conference.pdf) | NeurIPS  | RanPAC  | [GitHub](https://github.com/RanPAC/RanPAC)  | 2023 |
| 05  | [Few-shot class-incremental learning for medical time series classification](https://drive.google.com/file/d/1hsaJUvUPMAcMHuAoqL0wKUssG_o9rR71/view) | J-BHI   | MAPIC | - | 2023 |
| 06  | [Gkeal: Gaussian kernel embedded analytic learning for few-shot class incremental task](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhuang_GKEAL_Gaussian_Kernel_Embedded_Analytic_Learning_for_Few-Shot_Class_Incremental_CVPR_2023_paper.pdf) | CVPR | GKEAL  | -  | 2023 |
| 07 | [Self-regulating prompts: Foundational model adaptation without forgetting](https://openaccess.thecvf.com/content/ICCV2023/papers/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.pdf ) | ICCV | PromptSR | [GitHub](https://github.com/muzairkhattak/PromptSRC) | 2023 |

###### Feature Augmentation 2022 and earlier
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Self-promoted prototype refinement for few-shot class-incremental learning]( https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Self-Promoted_Prototype_Refinement_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf) | CVPR   | SPPR | -| 2021 |
| 02  | [Prototype augmentation and self-supervision for incremental learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf)|CVPR | PASS| [GitHub](https://github.com/Impression2805/CVPR21_PASS)| 2021 |
| 03  | [Memory-efficient incremental learning through feature adaptation](https://link.springer.com/chapter/10.1007/978-3-030-58517-4_41 )| ECCV |Feature_Adaptation | -  | 2020 |

## 2. Model Based Approaches 
### 2.1 Architecture-Based Approaches 
#### 2.1.1 Attention-based
###### Attention-based 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [SEP: Self-Enhanced Prompt Tuning for Visual-Language Model](https://arxiv.org/pdf/2405.15549) |arXiv| SEP| [GitHub](https://github.com/htyao89/SEP/) | 2024 |
| 02  | [Semantic Alignment for Prompt-Tuning in Vision Language Models](https://openreview.net/pdf?id=avDr56QjSI) | TMLR  | SAP | -  | 2024 |
| 03  | [Revisiting prompt pretraining of vision-language models](https://arxiv.org/pdf/2409.06166) | arXiv | RPP | - | 2024 |
| 04 | [Few-shot class incremental learning with attention-aware self-adaptive prompt](https://link.springer.com/chapter/10.1007/978-3-031-73004-7_1) | ECCV         | ASP | [GitHub](https://github.com/DawnLIU35/FSCIL-ASP) | 2024 |
|05| [M2sd: Multiple mixing self-distillation for few-shot class-incremental learning](https://ojs.aaai.org/index.php/AAAI/article/view/28129) | AAAI| M2SD | - |2024|
|06| [HPT++: Hierarchically Prompting Vision-Language Models with Multi-Granularity Knowledge Generation and Improved Structure Modeling](https://arxiv.org/pdf/2408.14812) | arXiv | HPT++  | - | 2024 |

###### Attention-based 2023 and earlier
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
|01| [Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) | CVPR | CODA-Prompt | [GitHub](https://github.com/GT-RIPL/CODA-Prompt)  | 2023 |
|02| [Dpl: Decoupled prompt learning for vision-language models](https://arxiv.org/pdf/2308.10061) | arXiv | DPL | -   | 2023 |
|03| [Continual learning with lifelong vision transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Continual_Learning_With_Lifelong_Vision_Transformer_CVPR_2022_paper.pdf) | CVPR  | LVT  | - | 2022 |

#### 2.1.2 Dynamic network structure-based
###### Dynamic network structure-based 2025
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
|01|[Skip tuning: Pre-trained vision-language models are effective and efficient adapters themselves](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Skip_Tuning_Pre-trained_Vision-Language_Models_are_Effective_and_Efficient_Adapters_CVPR_2025_paper.pdf) | CVPR | Skip_Tuning| [GitHub](https://github.com/Koorye/SkipTuning)| 2025 |
|02| [Continuous Knowledge-Preserving Decomposition for Few-Shot Continual Learning](https://arxiv.org/pdf/2501.05017) | arXiv | CKPD-FSCIL| [GitHub](https://github.com/xiaojieli0903/CKPD-FSCIL) | 2025 |

###### Dynamic network structure-based 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
|01|[Boosting continual learning of vision-language models via mixture-of-experts adapters](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters_CVPR_2024_paper.pdf) | CVPR | DDAS| [GitHub](https://github.com/JiazuoYu/MoE-Adapters4CL) | 2024 |
|02| [APLe: Token-Wise Adaptive for Multi-Modal Prompt Learning](https://arxiv.org/pdf/2401.06827) | arXiv  | APLe| -  | 2024 |
|03| [Revisiting prompt pretraining of vision-language models](https://arxiv.org/pdf/2409.06166) | arXiv | RPP  | -  | 2024 |

###### Dynamic network structure-based 2023
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 |  [On the Soft-Subnetwork for Few-Shot Class Incremental Learning](https://arxiv.org/pdf/2209.07529) | arXiv | SoftNet | [GitHub](https://github.com/ihaeyong/SoftNet-FSCIL) | 2023 |
| 02  |[Continual Learning via Winning Subnetworks That Arise Through Stochastic Local Competition](https://openreview.net/pdf?id=fUwfjPzI8g) | ICLR | TWTA-CIL      | -  | 2023 |
| 03  | [Domain incremental lifelong learning in an open world](https://arxiv.org/pdf/2305.06555) | arXiv | Diana | [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/diana) | 2023 |
| 04  |[Prompts can play lottery tickets well: Achieving lifelong information extraction via lottery prompt tuning](https://aclanthology.org/2023.acl-long.16.pdf) | ACL | LPT| [GitHub](https://github.com/jokieleung/Lottery_Prompt)  | 2023 |
| 05  | [Orthogonal subspace learning for language model continual learning](https://arxiv.org/pdf/2310.14152) | arXiv | O-LoRA | [GitHub](https://github.com/cmnfriend/O-LoRA)| 2023 |
| 06  |[Continual diffusion: Continual customization of text-to-image diffusion with c-lora](https://arxiv.org/pdf/2304.06027) | arXiv | C-LoRA| [GitHub](https://jamessealesmith.github.io/continual-diffusion/)  | 2023 |
| 07  | [Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) | CVPR  | CODA-Prompt| [GitHub](https://github.com/GT-RIPL/CODA-Prompt)   | 2023 |


###### Dynamic network structure-based 2022 and earlier
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Varigrow: Variational architecture growing for task-agnostic continual learning based on bayesian novelty](https://proceedings.mlr.press/v162/ardywibowo22a/ardywibowo22a.pdf) | ICML  | VariGrow | -  | 2022 |
| 02  | [Continual prompt tuning for dialog state tracking](https://arxiv.org/pdf/2203.06654) | arXiv | DST | [GitHub](https://github.com/thu-coai/CPT4DST)  | 2022 |
| 03  |[Dualprompt: Complementary prompting for rehearsal-free continual learning](https://link.springer.com/chapter/10.1007/978-3-031-19809-0_36) | ECCV | DualPrompt  | [GitHub](https://github.com/google-research/l2p)  | 2022 |
| 04  |[Der: Dynamically expandable representation for class incremental learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_DER_Dynamically_Expandable_Representation_for_Class_Incremental_Learning_CVPR_2021_paper.pdf) | CVPR         | DER  | [GitHub](https://github.com/Rhyssiyan/DER-ClassIL.pytorch) | 2021 |
| 05  |[Continual learning in task-oriented dialogue systems](https://arxiv.org/pdf/2012.15504) | arXiv | ToDs   | -  | 2020 |


### 2.2 Parameter-Space-Based Approaches 
#### 2.2.1 Feature Space
###### Feature Space 2025
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
|01|[Contrastive augmented graph2graph memory interaction for few shot continual learning](https://ieeexplore.ieee.org/abstract/document/10841449/) | TCSVT | LGP | - | 2025 |

###### Feature Space 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Expandable subspace ensemble for pre-trained model-based class-incremental learning](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Expandable_Subspace_Ensemble_for_Pre-Trained_Model-Based_Class-Incremental_Learning_CVPR_2024_paper.pdf) | CVPR | EASE | [GitHub](https://github.com/sun-hailong/CVPR24-Ease) | 2024 |
| 02 | [Rethinking misalignment in vision-language model adaptation from a causal perspective](https://proceedings.neurips.cc/paper_files/paper/2024/file/453a27b717972ef94a9a9113d236ad2f-Paper-Conference.pdf) | NeurIPS | CDC | [NeurIPS Code Policy](https://nips.cc/public/guides/CodeSubmissionPolicy) | 2024 |
| 03 |[DePT: Decoupled prompt tuning](http://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_DePT_Decoupled_Prompt_Tuning_CVPR_2024_paper.pdf) | CVPR | DePT | [GitHub](https://github.com/Koorye/DePT) | 2024 |
| 04 | [Delve into base-novel confusion: Redundancy exploration for few-shot class-incremental learning](https://arxiv.org/pdf/2405.04918) | arXiv | RDI | - | 2024 |
| 05 | [Deep Correlated Prompting for Visual Recognition with Missing Modalities](https://proceedings.neurips.cc/paper_files/paper/2024/file/7ca55c8276acf1f0aa996cd3622d1df4-Paper-Conference.pdf) | NeurIPS | DCP | [GitHub](https://github.com/hulianyuyy/Deep_Correlated_Prompting) | 2024 |
| 06 |[Craft: Cross-modal Aligned Features Improve Robustness of Prompt Tuning](https://arxiv.org/pdf/2407.15894) | arXiv | CRAFT | [GitHub](https://github.com/Jingchensun/Craft) | 2024 |
| 07 |[M2sd: Multiple mixing self-distillation for few-shot class-incremental learning](https://ojs.aaai.org/index.php/AAAI/article/view/28129) | AAAI | M2SD | - | 2024 |


###### Feature Space 2023
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 |[Ranpac: Random projections and pre-trained models for continual learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/2793dc35e14003dd367684d93d236847-Paper-Conference.pdf) | NeurIPS | RanPAC | [GitHub](https://github.com/RanPAC/RanPAC) | 2023 |
| 02 |[Big-model Driven Few-shot Continual Learning](https://arxiv.org/pdf/2309.00862) | arXiv | B-FSCL | - | 2023 |

###### Feature Space 2022
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Continual few-shot relation learning via embedding space regularization and data augmentation](https://arxiv.org/pdf/2203.02135) | arXiv | CFRL | [GitHub](https://github.com/qcwthu/Continual_Fewshot_Relation_Learning) | 2022 |
| 02 | [Learning to decompose visual features with latent textual prompts](https://arxiv.org/pdf/2210.04287) | arXiv | DeFo | - | 2022 |
| 03 |[Constrained few-shot class-incremental learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Hersche_Constrained_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf) | CVPR | C-FSCIL | [GitHub](https://github.com/IBM/constrained-FSCIL) | 2022 |
| 04 |[Few-shot class incremental learning leveraging self-supervised features](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Ahmad_Few-Shot_Class_Incremental_Learning_Leveraging_Self-Supervised_Features_CVPRW_2022_paper.pdf) | CVPR | FeSSSS | [GitHub](https://github.com/TouqeerAhmad/FeSSSS) | 2022 |

###### Feature Space 2021 and earlier
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 | [Self-promoted prototype refinement for few-shot class-incremental learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Self-Promoted_Prototype_Refinement_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf) | CVPR | SPPR | - | 2021 |
| 02 |[Online fast adaptation and knowledge accumulation (osaka): a new approach to continual learning](https://proceedings.neurips.cc/paper_files/paper/2020/file/c0a271bc0ecb776a094786474322cb82-Paper.pdf) | NeurIPS | OSAKA | [GitHub](https://github.com/ElementAI/osaka) | 2020 |


#### 2.2.2 Weight Space
###### Weight Space 2025
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01 |[Complementary Subspace Low-Rank Adaptation of Vision-Language Models for Few-Shot Classification](https://arxiv.org/pdf/2501.15040) | arXiv | Comp-LoRA | - | 2025 |
| 02 |[Efficient Few-Shot Continual Learning in Vision-Language Models](https://arxiv.org/pdf/2502.04098) | arXiv | LoRSU | - | 2025 |
| 03 |[Continuous Knowledge-Preserving Decomposition for Few-Shot Continual Learning](https://arxiv.org/pdf/2501.05017) | arXiv | CKPD-FSCIL | [GitHub](https://github.com/xiaojieli0903/CKPD-FSCIL) | 2025 |
| 04 |[Singular Value Fine-tuning for Few-Shot Class-Incremental Learning](https://arxiv.org/pdf/2503.10214) | arXiv | SVFCL | - | 2025 |

###### Weight Space 2024 and earlier
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 02 |[LW2G: Learning Whether to Grow for Prompt-based Continual Learning](https://openreview.net/pdf?id=QZuZmfLLRG) | ICLR | LW2G | [GitHub](https://github.com/thu-ml/HiDe-Prompt) | 2024 |
| 03 |[Continual Few-shot Relation Extraction via Adaptive Gradient Correction and Knowledge Decomposition](https://aclanthology.org/2024.findings-acl.702.pdf) | ACL | ROD_AOD | - | 2024 |
| 01 |[Warping the space: Weight space rotation for class-incremental few-shot learning](https://openreview.net/pdf?id=kPLzOfPfA2l) | ICLR | WaRP | [GitHub](https://github.com/EdwinKim3069/WaRP-CIFSL) | 2023 |


#### 2.2.3 Knowledge Distillation
###### Knowledge Distillation 2025
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Fully Fine-Tuning Beats Parameter Efficient Fine-Tuning for CLIP in Data-Limited Scenarios](https://openreview.net/pdf?id=VbszSB4pK6) | ICLR | CLIP-CITE      | - | 2025 |

###### Knowledge Distillation 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Improving Continual Few-shot Relation Extraction through Relational Knowledge Distillation and Prototype Augmentation](https://aclanthology.org/2024.lrec-main.767.pdf) | LREC-COLING  | RK2DA          | -                                                                 | 2024 |
| 02  |[On Distilling the Displacement Knowledge for Few-Shot Class-Incremental Learning](https://arxiv.org/pdf/2412.11017) | arXiv| DKD  | - | 2024 |
| 03  |[M2sd: Multiple mixing self-distillation for few-shot class-incremental learning](https://ojs.aaai.org/index.php/AAAI/article/view/28129) | AAAI         | M2SD | -  | 2024 |
| 04  |[Revisiting prompt pretraining of vision-language models](https://arxiv.org/pdf/2409.06166) | arXiv | RPP   | -  | 2024 |
| 05  |[Improving zero-shot generalization of learned prompts via unsupervised knowledge distillation](https://link.springer.com/chapter/10.1007/978-3-031-72907-2_27) |ECCV| KDPL| [GitHub](https://github.com/miccunifi/KDPL) | 2024 |
| 06  |[Cascade prompt learning for vision-language model adaptation](https://arxiv.org/pdf/2409.17805) | ECCV  | CasPL  | [GitHub](https://github.com/megvii-research/CasPL) | 2024 |

###### Knowledge Distillation 2023 and earlier
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Serial contrastive knowledge distillation for continual few-shot relation extraction](https://arxiv.org/pdf/2305.06616) | arXiv  | SCKD  | [GitHub](https://github.com/nju-websoft/SCKD) | 2023 |
| 02  | [Feature distribution distillation-based few shot class incremental learning](https://ieeexplore.ieee.org/abstract/document/9904282) | PRAI| - | -| 2022 |
| 03  |[Knowledge Distillation: A Survey](https://arxiv.org/pdf/2006.05525)   | arXiv | Survey | - | 2021 |



## 3. Optimization Based Approaches  
### 3.1 Gradient-Based Approaches  
#### 3.1.1 Function Regularization 
###### Function Regularization 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[HPT++: Hierarchically Prompting Vision-Language Models with Multi-Granularity Knowledge Generation and Improved Structure Modeling](https://arxiv.org/pdf/2408.14812) | arXiv | HPT++  | - | 2024 |
| 02  | [RESTORE: Towards Feature Shift for Vision-Language Prompt Learning](https://arxiv.org/pdf/2403.06136) | arXiv | RESTORE | [GitHub](https://github.com/Yaphabates/RESTORE_) | 2024 |
| 03  | [Conceptual Codebook Learning for Vision-Language Models](https://link.springer.com/chapter/10.1007/978-3-031-72980-5_14) | ECCV | CoCoLe| -  | 2024 |
| 04  |[Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models](https://arxiv.org/pdf/2410.02681) | arXiv | DOR| [GitHub](https://github.com/ml-stat-Sustech/Outlier-Calibration)  | 2024 |
| 05  |[Conditional prototype rectification prompt learning](https://ieeexplore.ieee.org/abstract/document/11069290/) | TCSVT | CPR | [GitHub](https://github.com/chenhaoxing/CPR) | 2024 |
| 06  |[Semantic Alignment for Prompt-Tuning in Vision Language Models](https://openreview.net/pdf?id=avDr56QjSI) | TMLR | SAP| - | 2024 |
###### Function Regularization 2023
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Self-regulating prompts: Foundational model adaptation without forgetting](https://openaccess.thecvf.com/content/ICCV2023/papers/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.pdf) | ICCV | PromptSR  | [GitHub](https://github.com/muzairkhattak/PromptSRC)| 2023 |
| 02  |[Towards Efficient Vision-Language Tuning: More Information Density, More Generalizability](https://arxiv.org/pdf/2312.10813) | arXiv| DIP| -| 2023 |


#### 3.1.2 Weight Regularization 
###### Weight Regularization 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [M³PL: Identifying and Exploiting View Bias of Prompt Learning](https://openreview.net/pdf?id=2rnTIBm19V) | Transact Mach Learn Res | M3PL | -  | 2024 |
| 02  | [Fine-Tuning CLIP's Last Visual Projector: A Few-Shot Cornucopia](https://inria.hal.science/hal-04986168/document) | Inria | -  | - | 2024 |
| 03  | [On Distilling the Displacement Knowledge for Few-Shot Class-Incremental Learning](https://arxiv.org/pdf/2412.11017) | arXiv | DKD   | -| 2024 |

###### Weight Regularization 2023
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) | CVPR  | CODA-Prompt    | [GitHub](https://github.com/GT-RIPL/CODA-Prompt) | 2023 |
| 02  |[Few shot class incremental learning via efficient prototype replay and calibration](https://www.mdpi.com/1099-4300/25/5/776) | Entropy| EPRC  | -  | 2023 |
| 03  |[Continual diffusion: Continual customization of text-to-image diffusion with c-lora](https://arxiv.org/pdf/2304.06027) | arXiv  | C-LoRA| [GitHub](https://jamessealesmith.github.io/continual-diffusion/)  | 2023 |

#### 3.1.3 Gradient space  
###### Gradient space  2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Generalizable Prompt Learning via Gradient Constrained Sharpness-aware Minimization](https://ieeexplore.ieee.org/abstract/document/10814656/) | TMM          | GCSCoOp | [GitHub](https://github.com/llcllc1997/GCSCoOp) | 2024 |
| 02 | [LW2G: Learning Whether to Grow for Prompt-based Continual Learning](https://openreview.net/pdf?id=QZuZmfLLRG) | ICLR | LW2G | [GitHub](https://github.com/thu-ml/HiDe-Prompt)  | 2024 |
| 03| [Continual Few-shot Relation Extraction via Adaptive Gradient Correction and Knowledge Decomposition](https://aclanthology.org/2024.findings-acl.702.pdf) | ACL | ROD&AOD | - | 2024 |
| 04  | [Fine-Tuning CLIP's Last Visual Projector: A Few-Shot Cornucopia](https://inria.hal.science/hal-04986168/document) | Inria   | -  | -  | 2024 |

###### Gradient space  2023
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Prompt gradient projection for continual learning](https://openreview.net/pdf?id=EH2O3h7sBI) | ICLR | PGP | [GitHub](https://github.com/JingyangQiao/prompt-gradient-projection) | 2023 |
| 02  |[Prompt-aligned gradient for prompt tuning](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Prompt-aligned_Gradient_for_Prompt_Tuning_ICCV_2023_paper.pdf) | ICCV | ProGrad  | - | 2023 |
| 03  |[Visual-language prompt tuning with knowledge-guided context optimization](https://openaccess.thecvf.com/content/CVPR2023/papers/Yao_Visual-Language_Prompt_Tuning_With_Knowledge-Guided_Context_Optimization_CVPR_2023_paper.pdf) | CVPR | KgCoOp  | -  | 2023 |

###### Gradient space  2021
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Gradient projection memory for continual learning](https://arxiv.org/pdf/2103.09762) | arXiv| GPM | [GitHub](https://github.com/sahagobinda/GPM) | 2021 |
| 02  |[Layerwise optimization by gradient decomposition for continual learning](http://openaccess.thecvf.com/content/CVPR2021/papers/Tang_Layerwise_Optimization_by_Gradient_Decomposition_for_Continual_Learning_CVPR_2021_paper.pdf) | CVPR | -  | -  | 2021 |

#### 3.1.4 Loss Function
###### Loss Function  2025
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Fully Fine-Tuning Beats Parameter Efficient Fine-Tuning for CLIP in Data-Limited Scenarios](https://openreview.net/pdf?id=VbszSB4pK6) | ICLR | CLIP-CITE  | - | 2025 |
| 02  |[Style-Pro: Style-Guided Prompt Learning for Generalizable Vision-Language Models](https://ieeexplore.ieee.org/abstract/document/10943992/) | WACV  | Style-Pro | - | 2025 |

###### Loss Function  2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Flatten long-range loss landscapes for cross-domain few-shot learning](https://openaccess.thecvf.com/content/CVPR2024/papers/Zou_Flatten_Long-Range_Loss_Landscapes_for_Cross-Domain_Few-Shot_Learning_CVPR_2024_paper.pdf) | CVPR  | FLoR | [GitHub](https://github.com/Zoilsen/FLoR) | 2024 |
| 02  |[A bag of tricks for few-shot class-incremental learning](https://arxiv.org/pdf/2403.14392) | arXiv | -  | - | 2024 |
| 03  |[Craft: Cross-modal Aligned Features Improve Robustness of Prompt Tuning](https://arxiv.org/pdf/2407.15894) | arXiv| CRAFT | [GitHub](https://github.com/Jingchensun/Craft) | 2024 |
| 04  |[Pre-trained vision and language transformers are few-shot incremental learners](https://openaccess.thecvf.com/content/CVPR2024/papers/Park_Pre-trained_Vision_and_Language_Transformers_Are_Few-Shot_Incremental_Learners_CVPR_2024_paper.pdf) | CVPR | PriViLege| [GitHub](https://github.com/KHU-AGI/PriViLege) | 2024 |
| 05  |[One-stage prompt-based continual learning](https://link.springer.com/chapter/10.1007/978-3-031-72624-8_10) | ECCV | PCL  | - | 2024 |
| 06  | [Conceptual Codebook Learning for Vision-Language Models](https://link.springer.com/chapter/10.1007/978-3-031-72980-5_14) | ECCV  | CoCoLe | - | 2024 |

###### Loss Function  2023
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 04  | [Few-shot class-incremental learning for medical time series classification](https://drive.google.com/file/d/1hsaJUvUPMAcMHuAoqL0wKUssG_o9rR71/view) | J-BHI | MAPIC | - | 2023 |
| 05  | [Ica-proto: Iterative cross alignment prototypical network for incremental few-shot relation classification](https://aclanthology.org/2023.findings-eacl.171.pdf) | EACL | ICA-Proto| -  | 2023 |
| 06  |[Consistent prototype learning for few-shot continual relation extraction](https://aclanthology.org/2023.acl-long.409.pdf) | ACL| ConPL | [GitHub](https://github.com/XiudiChen/ConPL)| 2023 |


###### Loss Function  2021
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Memory-efficient incremental learning through feature adaptation](https://link.springer.com/chapter/10.1007/978-3-030-58517-4_41) | ECCV |Feature_Adaptation | -  | 2020 |
| 02  |[Overcoming catastrophic forgetting in incremental few-shot learning by finding flat minima](https://proceedings.neurips.cc/paper_files/paper/2021/file/357cfba15668cc2e1e73111e09d54383-Paper.pdf) | NeurIPS  | F2M  | [GitHub](https://github.com/moukamisama/F2M) | 2021 |
| 03  |[Der: Dynamically expandable representation for class incremental learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_DER_Dynamically_Expandable_Representation_for_Class_Incremental_Learning_CVPR_2021_paper.pdf) | CVPR | DER | [GitHub](https://github.com/Rhyssiyan/DER-ClassIL.pytorch) | 2021 |

### 3.2 Tuning-based Approaches 
#### 3.2.1 Adapter-Based
###### Adapter-Based 2025
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[CMoA: Contrastive Mixture of Adapters for Generalized Few-Shot Continual Learning](https://ieeexplore.ieee.org/abstract/document/10891550/) | IEEE Transactions on Multimedia | CMoA | - | 2025 |
| 02  |[Continuous Knowledge-Preserving Decomposition for Few-Shot Continual Learning](https://arxiv.org/pdf/2501.05017) | arXiv| CKPD-FSCIL| [GitHub](https://github.com/xiaojieli0903/CKPD-FSCIL)| 2025 |
| 03  |[Complementary Subspace Low-Rank Adaptation of Vision-Language Models for Few-Shot Classification](https://arxiv.org/pdf/2501.15040) | arXiv| Comp-LoRA| -  | 2025 |
| 04  |[Singular Value Fine-tuning for Few-Shot Class-Incremental Learning](https://arxiv.org/pdf/2503.10214) | arXiv | SVFCL | - | 2025 |

###### Adapter-Based 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[RESTORE: Towards Feature Shift for Vision-Language Prompt Learning](https://arxiv.org/pdf/2403.06136) | arXiv | RESTORE| [GitHub](https://github.com/Yaphabates/RESTORE_)  | 2024 |
| 02  |[Conditional prototype rectification prompt learning](https://ieeexplore.ieee.org/abstract/document/11069290/) | TCSVT| CPR| [GitHub](https://github.com/chenhaoxing/CPR) | 2024 |
| 03  |[Boosting continual learning of vision-language models via mixture-of-experts adapters](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters_CVPR_2024_paper.pdf) | CVPR | DDAS | [GitHub](https://github.com/JiazuoYu/MoE-Adapters4CL) | 2024 |
| 04  |[Coin: A benchmark of continual instruction tuning for multimodel large language models](https://proceedings.neurips.cc/paper_files/paper/2024/file/6a45500d9eda640deed90d8a62742be5-Paper-Datasets_and_Benchmarks_Track.pdf) | NeurIPS | CoIN| [GitHub](https://github.com/zackschen/CoIN)  | 2024 |

###### Adapter-Based 2023 
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Continual diffusion: Continual customization of text-to-image diffusion with c-lora](https://arxiv.org/pdf/2304.06027) | arXiv| C-LoRA| [GitHub](https://jamessealesmith.github.io/continual-diffusion/)  | 2023 |
| 02  |[Towards Difficulty-Agnostic Efficient Transfer Learning for Vision-Language Models](https://arxiv.org/pdf/2311.15569) | arXiv| APEX  | - | 2023 |


###### Adapter-Based 2022 and earlier
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Lora: Low-rank adaptation of large language models](https://arxiv.org/pdf/2106.09685v1) | ICLR | LoRA| [GitHub](https://github.com/microsoft/LoRA)| 2022 |
| 02  |[Towards a unified view of parameter-efficient transfer learning](https://arxiv.org/pdf/2110.04366) | arXiv | - | [GitHub](https://github.com/jxhe/unify-parameter-efficient-tuning) | 2021 |


#### 3.2.2 Prompt-Based
###### Prompt-Based 2025
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Fully Fine-Tuning Beats Parameter Efficient Fine-Tuning for CLIP in Data-Limited Scenarios](https://openreview.net/pdf?id=VbszSB4pK6) | ICLR | CLIP-CITE| - | 2025 |

###### Prompt-Based 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
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
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Visual instruction tuning](https://proceedings.neurips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf) | NeurIPS          | LLaVA | [GitHub](https://llava-vl.github.io/) | 2023 |
| 02  | [Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) | CVPR | CODA-Prompt| [GitHub](https://github.com/GT-RIPL/CODA-Prompt) | 2023 |
| 03  |[MaPLe: Multi-modal prompt learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Khattak_MaPLe_Multi-Modal_Prompt_Learning_CVPR_2023_paper.pdf) | CVPR| MaPLe| [GitHub](https://github.com/muzairkhattak/multimodal-prompt-learning) | 2023 |
| 04  |[Self-regulating prompts: Foundational model adaptation without forgetting](https://openaccess.thecvf.com/content/ICCV2023/papers/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.pdf) | ICCV | PromptSR| [GitHub](https://github.com/muzairkhattak/PromptSRC) | 2023 |
| 05  | [Consistency-guided prompt learning for vision-language models](https://arxiv.org/pdf/2306.01195) | arXiv   | CoPrompt | [GitHub](https://github.com/ShuvenduRoy/CoPrompt) | 2023 |
| 06  |[Multimodal parameter-efficient few-shot class incremental learning](https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/DAlessandro_Multimodal_Parameter-Efficient_Few-Shot_Class_Incremental_Learning_ICCVW_2023_paper.pdf) | ICCV | CPE-CLIP | - | 2023 |

###### Prompt-Based 2022
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Learning to prompt for continual learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.pdf) | CVPR | L2P| [GitHub](https://github.com/google-research/l2p) | 2022 |
| 02  |[Dualprompt: Complementary prompting for rehearsal-free continual learning](https://link.springer.com/chapter/10.1007/978-3-031-19809-0_36) | ECCV| DualPrompt| [GitHub](https://github.com/google-research/l2p)| 2022 |
| 03| [Learning to prompt for vision-language models](https://arxiv.org/pdf/2109.01134) | IJCV| CoOp | [GitHub](https://github.com/KaiyangZhou/CoOp) | 2022 |
| 04  | [Visual prompt tuning](https://arxiv.org/pdf/2203.12119) | arXiv| VPT  | [GitHub](https://github.com/kmnp/vpt)  | 2022 |

###### Prompt-Based 2021
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Finetuned language models are zero-shot learners](https://arxiv.org/pdf/2109.01652) | arXiv | FLAN  | [GitHub](https://github.com/google-research/flan) | 2021 |
| 02  |[The power of scale for parameter-efficient prompt tuning](https://arxiv.org/pdf/2104.08691) | arXiv | - | [GitHub](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L264) | 2021 |
| 03  |[Bloom: A 176b-parameter open-access multilingual language model](https://arxiv.org/pdf/2211.05100) | arXiv | BLOOM| -| 2021 |

#### 3.2.3 Instruct Tuning
###### Instruct Tuning 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[InsCL: A data-efficient continual learning paradigm for fine-tuning large language models with instructions](https://arxiv.org/abs/2403.11435) | arXiv| InsCL| -| 2024 |
| 02  |[Multi-Task Transfer Matters During Instruction-Tuning](https://aclanthology.org/2024.findings-acl.883.pdf) | ACL| ICL| [GitHub](https://github.com/davidandym/Multitask-Transfer-Instruction-Tuning) | 2024 |
| 03  |[Coin: A benchmark of continual instruction tuning for multimodel large language models](https://proceedings.neurips.cc/paper_files/paper/2024/file/6a45500d9eda640deed90d8a62742be5-Paper-Datasets_and_Benchmarks_Track.pdf) | NeurIPS| CoIN | [GitHub](https://github.com/zackschen/CoIN) | 2024 |
| 04  |[Instruction-tuned language models are better knowledge learners](https://arxiv.org/pdf/2402.12847) | arXiv | PIT | - | 2024 |
| 05  | [Pllama: An open-source large language model for plant science](https://arxiv.org/pdf/2401.01600) | arXiv  | PLLaMa | [GitHub](https://github.com/Xianjun-Yang/PLLaMa) | 2024 |
| 06  |[Learning to poison large language models during instruction tuning](https://arxiv.org/pdf/2305.00944v1) | arXiv | - | [GitHub](https://github.com/AlexWan0/Poisoning-Instruction-Tuned-Models) | 2024 |
| 07  |[Continual LLaVA: Continual instruction tuning in large vision-language models](https://arxiv.org/pdf/2411.02564) | arXiv | Continual_LLaVA | [GitHub](https://github.com/mengcaopku/Continual-LLaVA) | 2024 |
| 08  |[Simple and scalable strategies to continually pre-train large language models](https://arxiv.org/pdf/2403.08763) | arXiv | LR_re-decaying | [GitHub](https://github.com/EleutherAI/gpt-neox) | 2024 |
| 09  |[Balancing Continuous Pre-Training and Instruction Fine-Tuning: Optimizing Instruction-Following in LLMs](https://arxiv.org/pdf/2410.10739) | arXiv| -  | - | 2024 |
| 10  |[Fine-tuning large language models with sequential instructions](https://arxiv.org/pdf/2403.07794) | arXiv  | SeqEval  | [GitHub](https://seqit.github.io/) | 2024 |
| 11  |[Self-Guide: Better task-specific instruction following via self-synthetic finetuning](https://arxiv.org/pdf/2407.12874) | arXiv | SELF-GUIDE| [GitHub](https://github.com/zhaochenyang20/Prompt2Model-Self-Guide)   | 2024 |

###### Instruct Tuning 2023
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Visual instruction tuning](https://proceedings.neurips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf) | NeurIPS | LLaVA| [GitHub](https://llava-vl.github.io/) | 2023 |
| 02  | [Dynosaur: A dynamic growth paradigm for instruction-tuning data curation](https://arxiv.org/pdf/2305.14327) | arXiv | DYNOSAUR  | [GitHub](https://github.com/WadeYin9712/Dynosaur)  | 2023 |
| 03  |[Conpet: Continual parameter-efficient tuning for large language models](https://arxiv.org/pdf/2309.14763) | arXiv| CONPET| [GitHub](https://github.com/Raincleared-Song/ConPET)| 2023 |
| 04  |[Orthogonal subspace learning for language model continual learning](https://arxiv.org/pdf/2310.14152) | arXiv| O-LoRA| [GitHub](https://github.com/cmnfriend/O-LoRA) | 2023 |

###### Instruct Tuning 2022 and earlier
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Fine-tuned language models are continual learners](https://arxiv.org/pdf/2205.12393) | arXiv | CT0 | [GitHub](https://github.com/ThomasScialom/T0_continual_learning)     | 2022 |
| 02  |[Finetuned language models are zero-shot learners](https://arxiv.org/pdf/2109.01652) | arXiv| FLAN  | [GitHub](https://github.com/google-research/flan)| 2021 |

#### 3.2.4 Prefix-Tuning
###### Prefix-Tuning 2024
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  | [Pre-trained vision and language transformers are few-shot incremental learners](https://openaccess.thecvf.com/content/CVPR2024/papers/Park_Pre-trained_Vision_and_Language_Transformers_Are_Few-Shot_Incremental_Learners_CVPR_2024_paper.pdf) | CVPR | PriViLege | [GitHub](https://github.com/KHU-AGI/PriViLege) | 2024 |
| 02  |[Q-tuning: Queue-based prompt tuning for lifelong few-shot language learning](https://arxiv.org/pdf/2404.14607) | arXiv| Q-Tuning | -   | 2024 |

###### Prefix-Tuning 2023 and earlier
| No. | Title |Venue | Algorithm Name | Code | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| 01  |[Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) | CVPR | CODA-Prompt| [GitHub](https://github.com/GT-RIPL/CODA-Prompt)| 2023 |
| 02  |[Dualprompt: Complementary prompting for rehearsal-free continual learning](https://link.springer.com/chapter/10.1007/978-3-031-19809-0_36) | ECCV | DualPrompt     | [GitHub](https://github.com/google-research/l2p) | 2022 |
| 03  |[Towards a unified view of parameter-efficient transfer learning](https://arxiv.org/pdf/2110.04366) | arXiv  | -  | [GitHub](https://github.com/jxhe/unify-parameter-efficient-tuning) | 2021 |
| 04  |[Prefix-tuning: Optimizing continuous prompts for generation](https://arxiv.org/pdf/2101.00190) | arXiv | prefix-tuning  | - | 2021 |

## 4. Survey of Incremental Learning
### Survey of Incremental Learning 2025
| No. | Title   | Venue | Code | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|
| 01  |[Few-Shot Class-Incremental Learning for Classification and Object Detection: A Survey](https://ieeexplore.ieee.org/abstract/document/10840313) | TPAMI | -| 2025 |
| 02  |[Recent advances of foundation language models-based continual learning: A survey](https://dl.acm.org/doi/abs/10.1145/3705725) | ACM  | - | 2025 |

### Survey of Incremental Learning 2024
| No. | Title   | Venue | Code | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|
| 01  |[A survey on few-shot class-incremental learning](https://www.sciencedirect.com/science/article/pii/S0893608023006019) | Neural_Networks  | -  | 2023 |
| 02  |[Class-incremental learning: A survey](https://ieeexplore.ieee.org/abstract/document/10599804) | TPAMI | [GitHub](https://github.com/zhoudw-zdw/CIL_Survey/)| 2024 |
| 03  |[Investigating the terrain of class-incremental continual learning: A brief survey](https://link.springer.com/chapter/10.1007/978-981-97-7426-5_25) | ICCCT | -   | 2024 |
| 04  |[Few-shot learning based on deep learning: A survey](https://www.aimspress.com/article/doi/10.3934/mbe.2024029) | AIMS| -  | 2024 |
| 05  | [A comprehensive survey of continual learning: Theory, method and application](https://ieeexplore.ieee.org/abstract/document/10444954) | TPAMI | - | 2024 |
| 06  |[Survey of continuous deep learning methods and techniques used for incremental learning](https://www.sciencedirect.com/science/article/abs/pii/S0925231224003163) | NEUROCOMPUTING  | -  | 2024 |

### Survey of Incremental Learning 2023
| No. | Title   | Venue | Code | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|
| 01  |  [A survey of incremental transfer learning: combining peer-to-peer federated learning and domain incremental learning for multicenter collaboration](https://arxiv.org/pdf/2309.17192) | arXiv | [GitHub](https://github.com/YixingHuang/ITLsurvey)| 2023 |
| 02  | [A comprehensive survey of few-shot learning: Evolution, applications, challenges, and opportunities](https://arxiv.org/pdf/2205.06743v2) | arXiv | -| 2023 |
| 03  |[Incremental learning with neural networks for computer vision: a survey](https://dl.acm.org/doi/abs/10.1145/3582688) | ACM  | -  | 2023 |

### Survey of Incremental Learning 2022
| No. | Title   | Venue | Code | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|
| 02  |[Three types of incremental learning](https://www.nature.com/articles/s42256-022-00568-3.pdf) | Nature | [GitHub](https://github.com/GMvandeVen/continual-learning)| 2022 |
| 03  |[A continual learning survey: Defying forgetting in classification tasks](https://ieeexplore.ieee.org/abstract/document/9349197) | TPAMI| [GitHub](https://github.com/MATTDL/CLSURVEY) | 2022 |
| 04  |[Learning from few examples: A summary of approaches to few-shot learning](https://arxiv.org/pdf/2203.04291) | arXiv| [Papers with Code](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-2) | 2022 |
| 05  |[Class-incremental learning: survey and performance evaluation on image classification](https://ieeexplore.ieee.org/abstract/document/9915459) | TPAMI | [GitHub](https://github.com/mmasana/FACIL) | 2022 |

### Survey of Incremental Learning 2020
| No. | Title   | Venue | Code | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|
| 01  |[Generalizing from a few examples: A survey on few-shot learning](https://dl.acm.org/doi/abs/10.1145/3386252) | ACM | [GitHub](https://github.com/tata1661/FewShotPapers.git) | 2020 |


## 5. Other Works
### Papers
| No. | Title   | Venue | Code | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:-----:|:-------:|:----:|
| 01  |[SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection](https://openaccess.thecvf.com/content/CVPR2025/papers/Tran_SimLTD_Simple_Supervised_and_Semi-Supervised_Long-Tailed_Object_Detection_CVPR_2025_paper.pdf) | CVPR             | - | 2025 |
| 02  | [General federated class-incremental learning with lightweight generative replay](https://ieeexplore.ieee.org/abstract/document/10612802/) | TPAMI  | -  | 2024 |
| 03  | [Semantically-shifted incremental adapter-tuning is a continual ViTransformer](https://openaccess.thecvf.com/content/CVPR2024/papers/Tan_Semantically-Shifted_Incremental_Adapter-Tuning_is_A_Continual_ViTransformer_CVPR_2024_paper.pdf) | CVPR| [GitHub](https://github.com/HAIV-Lab/SSIAT) | 2024 |
| 04  |[Vision-language models for vision tasks: A survey](https://ieeexplore.ieee.org/abstract/document/10445007/) | TPAMI| [GitHub](https://github.com/jingyi0000/VLM_survey)| 2024 |
| 05  |[CLIP with generative latent replay: a strong baseline for incremental learning](https://arxiv.org/pdf/2407.15793) | arXiv  | [GitHub](https://github.com/aimagelab/mammoth)  | 2024 |
| 06  | [Task adaptive parameter sharing for multi-task learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Wallingford_Task_Adaptive_Parameter_Sharing_for_Multi-Task_Learning_CVPR_2022_paper.pdf) | CVPR | -  | 2022 |
| 07  | [Biological underpinnings for lifelong learning machines](https://www.nature.com/articles/s42256-022-00452-0) | Nature | - | 2022 |
| 08  | [Learning transferable visual models from natural language supervision](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf) | ICML | [GitHub](https://github.com/OpenAI/CLIP) | 2021 |
| 09  | [VinVL: Revisiting visual representations in vision-language models](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VinVL_Revisiting_Visual_Representations_in_Vision-Language_Models_CVPR_2021_paper.pdf) | CVPR| [GitHub](https://github.com/pzzhang/VinVL)  | 2021 |
| 10  | [Lifelong machine learning](https://www.cs.uic.edu/~liub/lifelong-machine-learning-draft.pdf) | Morgan&Claypool  | - | 2018 |
| 11  | [Deep residual learning for image recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | CVPR | -  | 2016 |
| 12  | [ImageNet classification with deep convolutional neural networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | NeurIPS | [Code](http://code.google.com/p/cuda-convnet/)| 2012 |









