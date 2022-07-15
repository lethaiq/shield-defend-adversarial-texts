# SHILELD - NeuralPatcher - Defending Against Adversarial Text Attacks
Repository of the paper "SHIELD: Defending Textual Neural Networks against Multiple Black-Box Adversarial Attacks with Stochastic Multi-Expert Patcher" accepted to ACL'22 [[pdf]](https://arxiv.org/abs/2011.08908)

## Instructions
Installed a customized version of the ```OpenAttack``` framework that enables extracting the generated adversarial examples. Great thanks to the ```OpenAttack``` framework that was originally available at https://github.com/thunlp/OpenAttack
```
cd OpenAttack
python -m pip install -e .
```
Train the base model (without SHIELD protection). This will save the base model to ```model.pt``` in the current folder.
```
python train_base.py
```

Load the base model, and add the SHILED protection. Train the new model and save to ```shield.pt``` in the current folder. Please make sure that the maximum length of inputs is the same with the base model for fair comparisons. By default we use tau temperature of 1.0 for training. Often decreasing the batch_size will help the training of SHILED with better robustness.
```
python train_shield.py
```

Evaluate the base model with ```OpenAttack``` framework.
```
python evaluate_base.py
```

Evaluate the SHILELD-enabled model with ```OpenAttack``` framework. Often, we use tau temperature of 0.01 or 0.001 for inference.
```
python evaluate_shield.py
```

Below are results we got on 100 randomly sampled examples from the test set of ```clickbait detection``` and ```subjectivity detection```. This results are different from ones in the paper as we do not use parameter-search to find the best tau temperature during training and inference. Please refer to the procedure in the paper for the best results. Atk % is Attack Success Rate (the lower the better).
<img width="890" alt="image" src="https://user-images.githubusercontent.com/13818722/162591999-8532468f-3008-41d5-978b-089c80d29894.png">

## Additional Results
SHILED on ```clickbait``` dataset with ```max_len``` of 128 with attacker ```TextFooler```.

#### Evaluation on Base Model
```
ACC: 0.9903125
F1: 0.9902790843524616
+======================================+
|               Summary                |
+======================================+
| Total Attacked Instances:  | 100     |
| Successful Instances:      | 28      |
| Attack Success Rate:       | 0.28    |
| Avg. Running Time:         | 0.21282 |
| Total Query Exceeded:      | 0       |
| Avg. Victim Model Queries: | 46.75   |
+======================================+
(array([0, 1]), array([49, 51]))
Origin accuracy 1.0
Adversarial accuracy 0.43
Attack Success Rate 0.28
```

#### Evaluation on SHIELD. SHILED defends very well against TextFooler.
```
ACC: 0.99
F1: 0.9899749373433584
+======================================+
|               Summary                |
+======================================+
| Total Attacked Instances:  | 100     |
| Successful Instances:      | 7       |
| Attack Success Rate:       | 0.07    |
| Avg. Running Time:         | 0.21507 |
| Total Query Exceeded:      | 0       |
| Avg. Victim Model Queries: | 47.94   |
+======================================+
(array([0, 1]), array([49, 51]))
Origin accuracy 0.99
Adversarial accuracy 0.5
Attack Success Rate 0.06060606060606061
```



## A note on training
- It is best that the NAS weight parameters converge to one-hot vector during training. For example:
```
SHIELD NAS Decision: [array([9.800224e-04, **9.982419e-01**, 7.780853e-04], dtype=float32), array([5.7525238e-05, 2.1913149e-04, **9.9972326e-01**], dtype=float32), array([3.141981e-04, 1.747739e-04, **9.995110e-01**], dtype=float32), array([*0.9790363* , 0.00216483, 0.01879892], dtype=float32), array([2.3515357e-04, **9.9967730e-01**, 8.7526649e-05], dtype=float32)]
```

## Citation
```
@article{le2022shield,
  title={SHIELD: Defending Textual Neural Networks against Multiple Black-Box Adversarial Attacks with Stochastic Multi-Expert Patcher},
  author={Thai Le and Noseong Park and Dongwon Lee},
  journal={60th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2022}
}
```
