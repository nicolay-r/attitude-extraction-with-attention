from rusentrel.classic.common import classic_common_callback_modification_func

CV_COUNT = 3

CV_NAME_PREFIX = u'cv_'


def classic_cv_common_callback_modification_func(callback):
    """
    This function describes configuration setup for all model callbacks.
    """
    classic_common_callback_modification_func(callback)
    callback.set_cancellation_acc_bound(0.981)
    callback.set_cancellation_f1_train_bound(0.85)
    callback.set_key_save_hidden_parameters(False)
    callback.set_key_stop_training_by_cost(True)

