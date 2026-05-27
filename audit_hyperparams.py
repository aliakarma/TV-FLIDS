import yaml
cfg = yaml.safe_load(open('config/fl_config.yaml'))
print('Local LR:', cfg['federated_learning']['local_lr'])
print('Local epochs:', cfg['federated_learning']['local_epochs'])
print('Fraction fit:', cfg['federated_learning']['fraction_fit'])
print('# Rounds:', cfg['federated_learning']['num_rounds'])
print()
print('All strategies use the same local_lr, local_epochs, and fraction_fit.')
print('Trust-specific params apply only to TVFLIDSStrategy.')
