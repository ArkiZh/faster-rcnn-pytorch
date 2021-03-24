PRINT_FLAG = False


def check_cuda_usage(msg, empty_cache=False, print_flag=True):
    import torch
    if empty_cache:
        torch.cuda.empty_cache()
    if print_flag and PRINT_FLAG:
        print("\n===={}\n"
              "    Cuda memory allocated: {}\n"
              "    Cuda max memory allocated: {}\n"
              "    Cuda memory cached: {}\n"
              "    Cuda max memory cached: {}".format(msg,
                                                      torch.cuda.memory_allocated() / 2 ** 30,
                                                      torch.cuda.max_memory_allocated() / 2 ** 30,
                                                      torch.cuda.memory_cached() / 2 ** 30,
                                                      torch.cuda.max_memory_cached() / 2 ** 30))
