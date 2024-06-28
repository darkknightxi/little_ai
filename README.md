# Introduction
<img src="littleai.jpg" height="500" width="1048">
littleai is a lightweight and flexible deep learning library for building and training deep learning models with very minimal code and allows for a lot of customization.
# Example Usage

## Setup
Run the following in the command line or at the beginning of a jupyter notebook:
`!git clone https://github.com/darkknightxi/little_ai.git`

Importing the library and a few other useful things
```python
import torch.nn as nn
from little_ai.little_ai import learner, datasets 
import torchvision.transforms.functional as TF
from datasets import load_dataset
from torcheval.metrics import MulticlassAccuracy
```

## Preparing the DataLoaders

```python
# Load a dataset from HF
dataset = load_dataset('mnist')

# Specify transforms
def transforms(b):
  b['image'] = [TF.to_tensor(o) for o in b['image']]
  return b
dataset = dataset.with_transform(transforms)

# Turn it into dls
dls = datasets.DataLoaders.from_dd(dataset,batch_size=64)

# Look at the data
xb, yb = next(iter(dls.train))
```

## Prepare the Model
Any good ol' PyTorch model works

```python
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten()
)
```

## Create and Fit the Learner
The heart of little_ai is the `Learner` class. It pulls together the data, model and loss function, and can be extended in all sorts of cool ways using callbacks. Here's a somewhat minimal example, training our model on this classification task and plotting some stats as we do so:


```python
# There are callbacks for all sorts of things; here are some common ones:
cbs = [
    learner.TrainCB(), # Handles the core steps in the training loop. Can be left out if using TrainLearner
    learner.DeviceCB(), # Handles making sure data and model are on the right device
    learner.MetricsCB(accuracy=MulticlassAccuracy()), # Keep track of any relevant metrics
    learner.ProgressCB(), # Displays metrics and loss during training, optionally plot=True for a pretty graph
]

# Nothing fancy for the loss function
loss_fn = nn.CrossEntropyLoss()

# The learner takes a model, dataloaders and loss function, plus some optional extras like a list of callbacks
learn = learner.Learner(model, dls, loss_fn, lr=0.1, cbs=cbs)

# And fit does the magic :)
learn.fit(3)
```



