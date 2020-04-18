#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import tensorflow as tf


sys.path.append('../../../')

from arekit.networks.context.architectures.contrib.att_ends_pcnn import AttentionEndsPCNN
from arekit.networks.context.configurations.contrib.att_ends_pcnn import AttentionEndsPCNNConfig
from arekit.evaluation.evaluators.two_class import TwoClassEvaluator

from experiments.rusentrel.rusentrel_io import RuSentRelNetworkIO
from experiments.rusentrel.context.model import ContextLevelTensorflowModel

from rusentrel.ctx_names import ModelNames
from rusentrel.engine import run_testing
from rusentrel.callback import CustomCallback

from rusentrel.classic.common import \
    classic_ctx_common_config_settings, \
    classic_common_callback_modification_func


def ctx_att_pcnn_custom_config(config):
    assert(isinstance(config, AttentionEndsPCNNConfig))
    config.modify_bags_per_minibatch(2)
    config.modify_weight_initializer(tf.contrib.layers.xavier_initializer())


def run_testing_att_pcnn(name_prefix=u'',
                         cv_count=1,
                         custom_config_func=ctx_att_pcnn_custom_config,
                         custom_callback_func=classic_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + ModelNames().AttEndsPCNN,
                create_network=AttentionEndsPCNN,
                create_config=AttentionEndsPCNNConfig,
                create_io=RuSentRelNetworkIO,
                cv_count=cv_count,
                create_model=ContextLevelTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_ctx_common_config_settings)


if __name__ == "__main__":

    run_testing_att_pcnn()