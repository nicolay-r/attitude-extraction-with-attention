#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../../')

from arekit.networks.context.architectures.rcnn import RCNN
from arekit.networks.context.configurations.rcnn import RCNNConfig
from arekit.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.networks.tf_helpers.sequence import CellTypes

from rusentrel.callback import CustomCallback
from rusentrel.engine import run_testing
from rusentrel.ctx_names import ModelNames

from experiments.rusentrel.rusentrel_io import RuSentRelNetworkIO
from experiments.rusentrel.context.model import ContextLevelTensorflowModel

from rusentrel.classic.common import \
    classic_ctx_common_config_settings, \
    classic_common_callback_modification_func


def ctx_rcnn_custom_config(config):
    assert(isinstance(config, RCNNConfig))
    config.modify_bags_per_minibatch(2)
    config.modify_cell_type(CellTypes.LSTM)
    config.modify_dropout_rnn_keep_prob(0.9)


def run_testing_rcnn(name_prefix=u'',
                     cv_count=1,
                     custom_config_func=ctx_rcnn_custom_config,
                     custom_callback_func=classic_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + ModelNames().RCNN,
                create_network=RCNN,
                create_config=RCNNConfig,
                cv_count=cv_count,
                create_io=RuSentRelNetworkIO,
                create_model=ContextLevelTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_ctx_common_config_settings)


if __name__ == "__main__":

    run_testing_rcnn()

