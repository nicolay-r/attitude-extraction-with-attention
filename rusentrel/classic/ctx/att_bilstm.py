#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys


sys.path.append('../../../')

from arekit.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.networks.context.architectures.contrib.att_hidden_p_zhou_bilstm import AttentionHiddenPZhouBiLSTM
from arekit.networks.context.configurations.contrib.att_hidden_p_zhou_bilstm import AttentionHiddenPZhouBiLSTMConfig
from arekit.networks.tf_helpers.sequence import CellTypes

from rusentrel.callback import CustomCallback
from rusentrel.engine import run_testing
from rusentrel.ctx_names import ModelNames

from experiments.rusentrel.rusentrel_io import RuSentRelNetworkIO
from experiments.rusentrel.context.model import ContextLevelTensorflowModel

from rusentrel.classic.common import \
    classic_ctx_common_config_settings, \
    classic_common_callback_modification_func


def ctx_att_bilstm_custom_config(config):
    assert(isinstance(config, AttentionHiddenPZhouBiLSTMConfig))
    config.modify_bags_per_minibatch(2)
    config.modify_hidden_size(128)
    config.modify_cell_type(CellTypes.LSTM)
    config.modify_dropout_rnn_keep_prob(0.9)


def run_testing_att_bilstm(
        cv_count=1,
        name_prefix=u'',
        custom_config_func=ctx_att_bilstm_custom_config,
        custom_callback_func=classic_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + ModelNames().AttHiddenPZhouBiLSTM,
                create_network=AttentionHiddenPZhouBiLSTM,
                create_config=AttentionHiddenPZhouBiLSTMConfig,
                create_io=RuSentRelNetworkIO,
                cv_count=cv_count,
                create_model=ContextLevelTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_ctx_common_config_settings)


if __name__ == "__main__":

    run_testing_att_bilstm()
