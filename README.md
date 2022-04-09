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

Load the base model, and add the SHILED protection. Train the new model and save to ```shield.pt``` in the current folder. Please make sure that the maximum length of inputs is the same with the base model for fair comparisons. By default we use tau temperature of 1.0 for training. Often decreasing the batch_size will help the training of SHILED.
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

Below are results we got on 100 randomly sampled examples from the test set of ```clickbait detection``` and ```subjectivity detection```. This results are different from ones in the paper as we do not use parameter-search to find the best tau temperature during training and inference. Please follow the procedure in the paper for fair comparisons. Atk % is Attack Success Rate (the lower the better).
<img width="890" alt="image" src="https://user-images.githubusercontent.com/13818722/162591999-8532468f-3008-41d5-978b-089c80d29894.png">

## Citation
```
@article{le2022shield,
  title={SHIELD: Defending Textual Neural Networks against Multiple Black-Box Adversarial Attacks with Stochastic Multi-Expert Patcher},
  author={Thai Le and Noseong Park and Dongwon Lee},
  journal={60th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2022}
}
```
