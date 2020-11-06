import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration
    if model == 'RRSNet_model':
        from .RRSNet_model import RRSNetModel as M
    elif model == 'RRSGAN_model':
        from .RRSGAN_model import RRSGANModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
