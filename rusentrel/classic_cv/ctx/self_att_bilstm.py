#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys


sys.path.append('../../../')

from rusentrel.classic.ctx.self_att_bilstm import run_testing_self_att_bilstm
from rusentrel.classic_cv.common import CV_COUNT, \
    classic_cv_common_callback_modification_func, \
    CV_NAME_PREFIX


if __name__ == "__main__":

    run_testing_self_att_bilstm(
        name_prefix=CV_NAME_PREFIX,
        cv_count=CV_COUNT,
        custom_callback_func=classic_cv_common_callback_modification_func)
