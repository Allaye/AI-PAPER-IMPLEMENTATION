import torch


def check_gpu():
    '''
    get information on your current gpu
    '''
    if torch.cuda.is_available():
        print(f'GPU is available')
        print(f'u are currently using gpu number {torch.cuda.current_device()}')
        print(f'u currently have {torch.cuda.device_count()} gpu')
        print(f'u are currently using {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print(f'GPU is not available, you are using your cpu!')
    return None

def clean_gpu_memory():
    '''
    clean gpu memory
    '''
    torch.cuda.empty_cache()
    return None
