import time
import torch

import sys
sys.path.append('.')
from pytorch_inspector import ParrallelHandler

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("fork")

    try:

        a_vec = torch.randn(50,50,requires_grad=True)
        c_prime = torch.randn(80,80,requires_grad=True)

        ph1 = ParrallelHandler((640,480), 20.0, 50, 120, 20.0)
        ph2 = ParrallelHandler((640,480), 20.0, 50, 120, 20.0)
        queue, contextes = ph1.track_tensor(0, {'a_vec':a_vec, 'c_prime':c_prime})

        # Create a counter variable
        counter = 0

        # Create an infinite loop in the main process
        while True:
            # Increment the counter by 1
            counter += 1
            #print(f'main thread:{counter}')
            # Generate a line with the counter value
            line = f"This is line {counter} from main process"
            # Loop through the processes list
            for ctx in contextes:
                # Call the write_line method of each process with the line
                array = torch.rand(512,1024).unsqueeze(0)
                array = torch.nn.Parameter(array).share_memory_()

                list_data = []
                list_data.append('a_vec')
                list_data.append(line)
                list_data.append(array)
                # Put the obj in the queue
                if queue.qsize() > 2:
                    pass
                else:
                    queue.put_nowait(list_data)

                array = torch.rand(512,1024).unsqueeze(0)
                array = torch.nn.Parameter(array).share_memory_()

                list_data = []
                list_data.append('c_prime')
                list_data.append(line)
                list_data.append(array)
                # Put the obj in the queue
                if queue.qsize() > 2:
                    pass
                else:
                    queue.put_nowait(list_data)

            # Sleep for 1 second
            time.sleep(0.01)
            if counter >= 200:
                break

        # stop the processes (not necessary)
        #ph.stop()
        print('done')
        # Wait for all processes to finish (they never will)
        #for context in contextes:
        #    context.join()
        print(f'processes:{len(contextes)}')

    except Exception as e:
        print(f'test_fork ex:{e}')
    

