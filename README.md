# Imitation-Learning-PyTorch
Basic Behavioural Cloning and DAgger Implementation in PyTorch


## Behavioural Cloning:
* Define your policy network model in `model.py`.
* Get appropriate states from environment. Here I am creating random episodes during training.
* Extract the expert action [here](https://github.com/nsidn98/Imitation-Learning-PyTorch/blob/f98fa7b006cfceb07c3edd496fe88fc46b34b076/imitation_algos.py#L71) from a `.txt` file or a pickle file or some function of states.
* Run `python imitation_algos.py`.

### Requirements:
* numpy
* pytorch
* tensorboardX
* tqdm

#### To Do
* Implement DAgger 
* Make it compatible with OpenAI gym environments
* Add supporting algorithms to train further with Reinforcement Learning
* Comments in the code :P

#### For tensorboard:
`tensorboard --logdir=runs --host localhost --port 8088`
