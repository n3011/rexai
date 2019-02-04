# rexAI
Play Google's T-rex game using Deep Q-Learning.

## Prerequisites
 - tensorflow >= 1.9.0
 - tefla

## Getting started

### Installation

```Shell
git clone https://github.com/n3011/rexai.git
cd rexai
pip install -r requirements.txt
export PYTHONPATH=.
```

### Webserver for running the javascript T-rex game

A simple webserver is required to run the T-rex javascript game.
```Shell
$ cd rextf/game
$ python2 -m SimpleHTTPServer <port>
```
The game is now accessable on your localhost `127.0.0.1:port`.


### Run training

```Shell
python main_loop.py --logdir /path/to/logir

```
Open your browswer and press `F5` to start the game.


## References
The game environment and some parts of the code are inspired from https://github.com/vdutor/TF-rex.
