import cv2
import time
from pygame import mixer

try:
    from time import time_ns
except ImportError:
    from time import time
    def time_ns(): return int(time() * 1e9)

from alga import PictureOutput, U8Picture, JsonInput

# Sonido de la alarma
mixer.init()
sound = mixer.Sound('Drowsiness detection/alarm.wav')

def run(outaddr, json_inaddr, slice='/', device='d', hwm=1,  multithreading=False):

    with PictureOutput(outaddr, f'{slice}video/', bind=False, multithreading=multithreading) as output, \
            JsonInput(json_inaddr, f'{slice}meta/', bind=False, hwm=hwm,multithreading=multithreading) as input_json:

            output.format = 'JPEG'
            packet = U8Picture()
            packet.topic = f'{slice}video/{device}/'
            packet.format = 'JPEG'

            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            conta = 10

            start_client = time.time()


            while conta >= 1:

                ret, frame = cap.read()
                cv2.imshow('frame', frame)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    
                packet.seq += 1
                packet.ts = time_ns()
                packet.data = frame

                output.send(packet)

                json_meta = input_json.recv()
                if not json_meta is None:
                    print("Car engine: ", json_meta.data)
                conta = conta - 1
            end_client = time.time()

            while True:

                ret, frame = cap.read()
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                packet.seq += 1
                packet.ts = time_ns()
                packet.data = frame

                output.send(packet)

                json_meta = input_json.recv()
                if not json_meta is None:
                    if(json_meta.data == 'Dormido'):
                        sound.play()
                        print("State: ", json_meta.data)
                end_frame = time.time()
            print("\nEl tiempo del cliente es de: ", end_client - start_client, "\n\n")


if __name__ == '__main__':

    outaddr = 'tcp://127.0.0.1:8000'
    json_inaddr = 'tcp://127.0.0.1:8060'


    slice = '/dr/'
    device = 'client'
    hwm = 1
    multithreading = False

    run(outaddr, json_inaddr, slice, device, hwm, multithreading)
    