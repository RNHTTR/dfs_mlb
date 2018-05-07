import yaml


# class Config:
#     '''
#     '''
#     def __init__
def read_config(config_file_name):
    '''
    '''
    with open(config_file_name) as config:
        return yaml.load(config)
