import os
import time
import numpy as np
GPU_MEMORY_NEEDED = 10000


def check_util_free(command_list):
    while True:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        os.system('rm tmp')

        most_free_gpu = int(np.argmax(memory_gpu))
        if memory_gpu[most_free_gpu] > GPU_MEMORY_NEEDED:
            for command in command_list:
                print(command.format(most_free_gpu))
                os.system(command.format(most_free_gpu))
            break
        time.sleep(60)



