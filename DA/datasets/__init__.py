from .visda17 import form_visda17
from .visda18 import form_visda18
from .domainnet import form_domainnet
from .office import form_office

# Known : unknown ratio
# ano_type = 1 -> 1:10 (competition)
# ano_type = 2 -> 1:1
# ano_type = 3 -> 10:1

def form_visda_datasets(config, ignore_anomaly=False):
    if config.dataset == 'VISDA17':
        return form_visda17(config)
    elif config.dataset == 'VISDA18':
        return form_visda18(config, ignore_anomaly, config.ano_type)
    elif config.dataset == 'DomainNet':
        return form_domainnet(config)
    elif config.dataset == 'office':
        return form_office(config)
    else:
        raise ValueError('Please specify a valid dataset | VISDA17 / VISDA18')