import cv2
import numpy as np
import os

try:
    from time import time_ns
except ImportError:
    from time import time
    def time_ns(): return int(time() * 1e9)

from alga import U8PictureInput, U8PictureOutput, PictureInput, PictureOutput, JsonInput, Picture, U8Picture

def run(inaddr, slice='/', device='d', hwm=1,  multithreading=False):
    with PictureInput(inaddr, f'{slice}arms/', bind=False, multithreading=multithreading) as input:

        while True:

            msg = input.recv()
            if not msg is None:
                image = msg.data
                print(image)
                #image = image.astype(np.uint8)
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('Faces', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == '__main__':

    inaddr = 'tcp://10.209.2.118:8090'
    slice = '/dr/'
    device = 'client2'
    hwm = 1
    multithreading = False
    run(inaddr, slice, device, hwm, multithreading)
