import base64
import json
import re
import time
import threading
import multiprocessing
from io import BytesIO
import numpy as np
from PIL import Image

from ..server.websocket_server import WebsocketServer


class Action:
  UP = 0
  DOWN = 1
  FORWARD = 2


class Environment(object):
  """
    Environment class is responsible for passing the actions to the game.
    It is also responsible for retrieving the game status and the reward.
    """

  def __init__(self, host, port, debug=False):
    self.debug = debug
    self.queue = multiprocessing.Queue()
    self.game_client = None
    self.server = WebsocketServer(port, host=host)
    self.server.set_fn_new_client(self.new_client)
    self.server.set_fn_message_received(self.new_message)
    print("\nGame can be connected (press F5 in Browser)")
    thread = threading.Thread(target=self.server.run_forever)
    thread.daemon = True
    thread.start()

  @property
  def actions(self):
    return {Action.UP: 'UP', Action.FORWARD: 'FORTH', Action.DOWN: 'DOWN'}

  @property
  def screen_width(self):
    return 80

  @property
  def screen_height(self):
    return 80

  def new_client(self, client, server):
    if self.debug: print("GameAgent: Game just connected")
    self.game_client = client
    self.server.send_message(self.game_client, "Connection to Game Agent Established")

  def new_message(self, client, server, message):
    if self.debug: print("GameAgent: Incoming data from game")
    data = json.loads(message)
    image, crashed = data['world'], data['crashed']

    # remove data-info at the beginning of the image
    image = re.sub('data:image/png;base64,', '', image)
    # convert image from base64 decoding to np array
    image = np.array(Image.open(BytesIO(base64.b64decode(image))))

    # cast to bool
    crashed = True if crashed in ['True', 'true'] else False

    self.queue.put((image, crashed))

  def start_game(self):
    """
        Starts the game and lets the TRex run for half a second and then returns the initial state.

        Returns:
            the initial state of the game (np.array, reward, crashed).
        """
    # game can not be started as long as the browser is not ready
    while self.game_client is None:
      time.sleep(1)

    self.server.send_message(self.game_client, "START")
    time.sleep(4)
    return self.get_state(Action.FORWARD)

  def refresh_game(self):
    time.sleep(0.5)
    print("...refreshing game...")
    self.server.send_message(self.game_client, "REFRESH")
    time.sleep(1)

  def do_action(self, action):
    """
        Performs action and returns the updated status
        
        Args:
            action:  Must come from the class Action.
                        The only allowed actions are Action.UP, Action.Down and Action.FORWARD.
        Retusn: 
            return the image of the game after performing the action, the reward (after the action) and
                        whether the TRex crashed or not.
        """
    if action != Action.FORWARD:
      # noting needs to be send when the action is going forward
      self.server.send_message(self.game_client, self.actions[action])

    time.sleep(.05)
    return self.get_state(action)

  def get_state(self, action):
    self.server.send_message(self.game_client, "STATE")

    image, crashed = self.queue.get()

    if crashed:
      reward = -100.
    else:
      if action == Action.UP:
        reward = -5.
      elif action == Action.DOWN:
        reward = -3.
      else:
        reward = 1.

    return image, reward, crashed
