#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import tensorflow as tf

sys.path.append('../../../')

from arekit.networks.context.architectures.contrib.ian_ef import IANEndsAndFrames
from arekit.networks.context.configurations.contrib.ian_ef import IANEndsAndFramesConfig
from arekit.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.networks.tf_helpers.sequence import CellTypes

from experiments.rusentrel.rusentrel_io import RuSentRelNetworkIO
from experiments.rusentrel.context.model import ContextLevelTensorflowModel

from rusentrel.callback import CustomCallback
from rusentrel.ctx_names import ModelNames
from rusentrel.engine import run_testing

from rusentrel.classic.common import \
    classic_ctx_common_config_settings, \
    classic_common_callback_modification_func


def ctx_ian_ef_custom_config(config):
    assert(isinstance(config, IANEndsAndFramesConfig))
    config.modify_bags_per_minibatch(2)
    config.modify_cell_type(CellTypes.LSTM)
    config.modify_l2_reg(0.0)
    config.modify_learning_rate(0.1)
    config.modify_weight_initializer(tf.contrib.layers.xavier_initializer())
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_gpu_memory_fraction(0.7)


def run_testing_ian_ef(cv_count=1,
                       name_prefix=u'',
                       custom_config_func=ctx_ian_ef_custom_config,
                       custom_callback_func=classic_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + ModelNames().IANEndsAndFrames,
                create_network=IANEndsAndFrames,
                create_config=IANEndsAndFramesConfig,
                create_io=RuSentRelNetworkIO,
                cv_count=cv_count,
                create_model=ContextLevelTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_ctx_common_config_settings)


if __name__ == "__main__":

    run_testing_ian_ef()
