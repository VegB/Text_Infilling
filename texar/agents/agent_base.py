#
"""
Base class for reinforcement learning agents.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.hyperparams import HParams
from texar.utils import utils

# pylint: disable=too-many-instance-attributes

class AgentBase(object):
    """
    Base class inherited by RL agents.

    Args:
        TODO
    """
    def __init__(self, env_config, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())
        self._env_config = env_config

        name = self._hparams.name
        self._variable_scope = utils.get_unique_named_variable_scope(name)
        self._unique_name = self.variable_scope.name.split("/")[-1]

        self._reset_tmplt_fn = tf.make_template(
            "{}_reset".format(self.name), self._reset)
        self._observe_tmplt_fn = tf.make_template(
            "{}_observe".format(self.name), self._observe)
        self._get_action_tmplt_fn = tf.make_template(
            "{}_get_action".format(self.name), self._get_action)

        self._timestep = 0

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        return {
            'name': 'agent'
        }

    def reset(self):
        """Resets the states to begin new episodes.
        """
        self._reset_tmplt_fn()

    def _reset(self):
        raise NotImplementedError

    def observe(self, reward, terminal, mode):
        """Observes experience from environment.

        Args:
        """
        return self._observe_tmplt_fn(reward, terminal, mode)

    def _observe(self, reward, terminal, mode):
        raise NotImplementedError

    def get_action(self, observ, mode, feed_dict=None):
        """Gets action according to observation.

        Args:

        Returns:
        """
        return self._get_action_tmplt_fn(observ, mode, feed_dict)

    def _get_action(self, observ, mode, feed_dict=None):
        raise NotImplementedError

    @property
    def variable_scope(self):
        """The variable scope of the agent.
        """
        return self.variable_scope

    @property
    def name(self):
        """The uniquified name of the module.
        """
        return self._unique_name

    @property
    def hparams(self):
        """A :class:`~texar.hyperparams.HParams` instance. The hyperparameters
        of the module.
        """
        return self._hparams
