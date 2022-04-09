# SHILELD - NeuralPatcher - Defending Against Adversarial Text Attacks
Repository of the paper "SHIELD: Defending Textual Neural Networks against Multiple Black-Box Adversarial Attacks with Stochastic Multi-Expert Patcher" accepted to ACL'22 [[pdf]](https://arxiv.org/abs/2011.08908)

## Instructions

Train the base model (without SHIELD protection). This will save the base model to ```model.pt``` in the current folder.
```
python train_base.py
```

Load the base model, and add the SHILED protection. Train the new model and save to ```shield.pt``` in the current folder. Please make sure that the maximum length of inputs is the same with the base model for fair comparisons. By default we use tau temperature of 1.0 for training.
```
python train_shield.py
```

Evaluate the base model with ```OpenAttack``` framework.
```
python evaluate_base.py
```

Evaluate the SHILELD-enabled model with ```OpenAttack``` framework. By default we use tau temperature of 0.01 for inference.
```
python evaluate_shield.py
```

Below are results we got on 100 randomly sampled examples from the test set of two datasets: ```clickbait detection``` and ```SST```. This results are different from ones in the paper as we do not use parameter-search to find the best tau temperature during training and inference. Please follow the procedure in the paper for fair comparisons.

```

```


