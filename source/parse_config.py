def parse_model_config(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x.strip() for x in lines if x and not x.startswith('#')]

    modules = []
    for line in lines:
        if line.startswith('['):
            modules.append({})
            modules[-1]['type'] = line[1:-1].strip()
            if modules[-1]['type'] == 'convolutional':
                modules[-1]['batch_normalize'] = 0
        else:
            key, val = line.split('=')
            modules[-1][key.strip()] = val.strip()
    return modules

def parse_data_config(path):
    options = {'gpus': '0,1,2,3', 'num_workers': '10'}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()
    return options
