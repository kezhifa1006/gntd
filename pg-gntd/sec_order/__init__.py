from sec_order import td_kfac
from sec_order import kfac, ekfac, td_kfac, lstd, adamw, kfac_ming
from sec_order import _utils, types


def get_optimizer(name):
    if name == 'kfac':
        return kfac.KFACOptimizer
    elif name == 'ekfac':
        return ekfac.EKFACOptimizer
    else:
        raise NotImplementedError