PRINT_FLAG = False


def check_cuda_usage(msg, empty_cache=False, print_flag=True):
    import torch
    if empty_cache:
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_max_memory_allocated()
    if print_flag and PRINT_FLAG:
        print("\n==== {}\n"
              "    Empty cache done: {}\n"
              "    Cuda memory allocated: {}\n"
              "    Cuda max memory allocated: {}\n"
              "    Cuda memory cached: {}\n"
              "    Cuda max memory cached: {}".format(msg,
                                                      str(empty_cache),
                                                      torch.cuda.memory_allocated() / 2 ** 30,
                                                      torch.cuda.max_memory_allocated() / 2 ** 30,
                                                      torch.cuda.memory_cached() / 2 ** 30,
                                                      torch.cuda.max_memory_cached() / 2 ** 30))
