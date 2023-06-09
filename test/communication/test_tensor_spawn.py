import time
import torch

import sys
sys.path.append('.')
from pytorch_inspector import ParrallelHandler, DataRecorder, ProcessInfoData

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")

    a_vec = torch.randn(50,50,requires_grad=True)
    c_prime = torch.randn(80,80,requires_grad=True)

    dr0 = DataRecorder(shape_expected=(640,480), fps=20., maxframes=30, path_root='output', 
                       colorBGR=(255,0,255), displayND_mode='default')
    ph1 = ParrallelHandler(callback_onrun=dr0.tensor_plot2D, callback_onclosing=dr0.flush, 
                           frequency=20.0, timeout=30.0, target_method='spawn', max_queue_size=1000, daemon=False)
    unique_id, queue_to, queue_from, contexts = ph1.track_tensor(0, {'a_vec':a_vec, 'c_prime':c_prime}, callback_transform=None)

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
        for ctx in contexts:
            # Call the write_line method of each process with the line
            array = torch.rand(512,1024).unsqueeze(0)
            array = torch.nn.Parameter(array).share_memory_()

            # Put the obj in the queue
            if queue_to.qsize() > 4:
                pass
            else:
                info_data = ProcessInfoData(name='a_vec', internal_message='internal_line', 
                                            message=line, shared_data=array)
                queue_to.put_nowait(info_data)

            array = torch.rand(512,1024).unsqueeze(0)
            array = torch.nn.Parameter(array).share_memory_()

            # Put the obj in the queue
            if queue_to.qsize() > 4:
                pass
            else:
                info_data = ProcessInfoData(name='c_prime', internal_message='internal_line', 
                                            message=line, shared_data=array)
                queue_to.put_nowait(info_data)

        # Sleep for 1 second
        time.sleep(0.01)
        if counter >= 200:
            break

    # Stop the processes. Since they are running as daemon, no join is done.
    ph1.stop(check_is_alive = True)