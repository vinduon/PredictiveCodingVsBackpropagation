# Predictive Coding Vs BackPropagation
Refactored code to accompany the paper "On the relationship between predictive coding and backpropagation"

Source: https://github.com/RobertRosenbaum/PredictiveCodingVsBackProp

## How to run
### Experiment 1
```commandline
python main.py --lr=0.1 --num_epoch=10 --opt=sgd --err_type=strict --model_type=original
```
### Experiment 2
```commandline
python main.py --lr=0.1 --num_epoch=10 --opt=sgd --err_type=fixed_pred --num_iter=5 --model_type=original --eta=1
```
### Experiment 3
```commandline
python main.py --lr=0.001 --num_epoch=5 --opt=adam --err_type=fixed_pred --num_iter=20 --model_type=modified --train_batch_size=1000 --test_batch_size=1000 
```
For other options, take a look at  ```main.py``` file.
