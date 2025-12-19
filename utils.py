import logging

def handle_txt(txt_path):
    '''
    Read txt file script
    input: 
          txt_path: .txt file's file path
    output: 
          txt_dict: {'key':'wave file path'}
    '''
    txt_dict = dict()
    line = 0
    lines = open(txt_path, 'r').readlines()
    for l in lines:
        txt_parts = l.strip().split()
        # print(len(txt_parts))
        line += 1
        if len(txt_parts) != 2:
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(
                txt_path, line, txt_parts))
        if len(txt_parts) == 2:
            key, value = txt_parts
        if key in txt_dict:
            raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                key, txt_path))

        txt_dict[key] = value

    return txt_dict


def get_logger(name, format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
               date_format='%Y-%m-%d %H:%M:%S', file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6

if __name__ == "__main__":
    print(len(handle_txt('/home/likai/data1/create_txt/cv_s2.txt')))
