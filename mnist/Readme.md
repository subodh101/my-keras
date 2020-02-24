# MNIST

A structured implementation of [keras/save_and_load](https://www.tensorflow.org/tutorials/keras/save_and_load)

```bash
├── dataset.py [get the dataset]
├── model.py [get the model]
├── predict.py [predict from the best saved weight]
└── train.py [train and save the best weight]
```

### WorkFlow

Clone the repo
```bash
git clone https://github.com/subodh101/my-keras.git
cd mnist/

```

Install pipenv and dependencies
```bash
pip install pipenv && pipenv install

```

Run the training
```bash
pipenv run python train.py
```

Run the prediction
```bash
pipenv run python predict.py
```

