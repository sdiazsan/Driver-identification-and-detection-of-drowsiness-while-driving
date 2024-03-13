import zmq
import json
import cv2
import numpy as np
import threading

MSG_VERSION = 1

PIC_FORMATS = ['JPEG', 'RGB', 'ENCODED', 'GRAY8'] # The only supported picture formats

_def_packet_h = {'ver': 1, 'seq': 0, 'ts': 0, 'size': 0, 'meta': None}
_def_picture_h = {'width': 1, 'height': 1, 'channels': 1, 'flip': False, 'format': 'GRAY8'}
_def_picture_h.update(_def_packet_h)

def jpeg_decode(array):
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    return img

def jpeg_encode(img, channels):
    if channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, img = cv2.imencode(".jpg", img)
    return img

def validate_format(fmt):
    if fmt not in PIC_FORMATS:
        raise ValueError(f'Format {fmt} not supported')

class Packet:
    def __init__(self, topic='/', h=_def_packet_h, payload=None):
        self.topic = topic
        self.ver = h['ver']
        self.seq = h['seq']
        self.ts = h['ts']
        self.size = h['size']
        try:
            self.meta = h['meta']
        except KeyError:
            self.meta = None
        self.payload = payload
        if self.ver != MSG_VERSION:
            raise RuntimeError(f'Bad version {self.ver}')

        self.topic_suffix = '' # To be filled on reception

    @property
    def metadata(self):
        return self.meta

    @metadata.setter
    def metadata(self, meta):
        self.meta = meta

    @property
    def header(self):
        return {'ver': self.ver, 'seq': self.seq, 'ts': self.ts, 'size': self.size, 'meta': self.meta}

    @property
    def data(self):
        return self.payload

    @data.setter
    def data(self, data):
        self.payload = data
        self.size = len(self.payload)


class Json(Packet):

    @property
    def data(self):
        return json.loads(self.payload.decode('utf-8'))

    @data.setter
    def data(self, data):
        self.payload = json.dumps(data).encode('utf-8')
        self.size = len(self.payload)

class Picture(Packet):
    def __init__(self, topic='/', h=_def_picture_h, payload=None, scale=255.):
        super().__init__(topic, h, payload)
        self.width = h['width']
        self.height = h['height']
        self.channels = h['channels']
        self.format = h['format']
        self.flip = h['flip']
        self.scale = scale
        validate_format(self.format)

    @property
    def header(self):
        header = super().header
        header.update({'width': self.width, 'height': self.height, 'channels': self.channels, 'flip': self.flip, 'format': self.format})

        return header

    @property
    def data(self):
        array = np.frombuffer(self.payload, dtype=np.uint8)
        # Decode from jpeg
        if self.format == 'JPEG':
            img = jpeg_decode(array)
        # If not encoding, reshape
        elif self.format == 'RGB' or self.format == 'GRAY8':
            img = array.reshape((self.height, self.width, self.channels))
        else:
            return array

        if self.scale:
            img = img / self.scale
        return img

    @data.setter
    def data(self, data):
        if self.format in ['JPEG', 'RGB', 'GRAY8']:
            if self.scale:
                data = data * self.scale
            try:
                self.height, self.width, self.channels = data.shape
            except ValueError:
                self.height, self.width = data.shape
                self.channels = 1

        img = np.array(data, dtype=np.uint8)
        if self.format == 'JPEG':
            img = jpeg_encode(img, self.channels)
        self.payload = img.tobytes()
        self.size = len(self.payload)


class U8Picture(Picture):
    def __init__(self, topic='/', h=_def_picture_h, payload=None):
        super().__init__(topic, h, payload, scale=None)

class IO:
    _context = None

    @classmethod
    def context(cls):
        if not cls._context:
            cls._context = zmq.Context()
        return cls._context

    def __init__(self, endpoint, topic, bind=True, hwm=None, socktype=zmq.PUB, packtype=Packet, multithreading = False):
        self.endpoint = endpoint
        self.topic = topic
        self.bind = bind
        self.hwm = hwm
        self.socktype = socktype
        self.packtype = packtype
        self.multithreading = multithreading and self.socktype is zmq.SUB
        self.pkt = None
        self.isPkt = False
        self.socket = IO.context().socket(socktype)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        if self.multithreading:
            self.timeout = 1
        else:
            self.timeout = 1000

        if socktype == zmq.SUB:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")#self.topic)
        if hwm is not None:
            self.socket.hwm = hwm
        #self.socket.setsockopt(zmq.RCVTIMEO, 200)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print([exc_type, exc_val, exc_tb])
        self.disconnect()

    def connect(self):
        if self.bind:
            self.socket.bind(self.endpoint)
        else:
            self.socket.connect(self.endpoint)
        if self.multithreading:
            #import time
            #time.sleep(1)
            self.t = threading.Thread(target = self.recvWorker)
            self.t.setDaemon(True)
            self.t.start()

    def disconnect(self):
        if(self.multithreading):
            self.threadFlag = False
            self.t.join()
        self.socket.close()
        print("Disconnected")

    def recvWorker(self):
        self.threadFlag = True
        while self.threadFlag:
            if self.poller.poll(self.timeout):
                self.pkt = self.recvSocket()
                self.isPkt = not self.pkt is None

    def recv(self):
        if self.multithreading:
            if self.isPkt:
                self.isPkt = False
                return self.pkt
            else:
                return None
        else:
            if self.poller.poll(self.timeout):
                return self.recvSocket()
            print("TIMEOUT")

    def recvSocket(self):
        topic, header, payload = self.socket.recv_multipart()
        topic = topic.decode('utf-8')
        header = json.loads(header.decode('utf-8'))
        packet = self.packtype(topic, header, payload)
        packet.topic_suffix = packet.topic[len(self.topic):]
        return packet

    def send(self, packet):
        packet.topic = self.topic.encode('utf-8')
        header = json.dumps(packet.header).encode('utf-8')
        self.socket.send_multipart([packet.topic , header, packet.payload])

    def response(self, packet):
        """
        Generate response to a packet
        :param packet:
        :return:
        """
        topic = self.topic + packet.topic_suffix
        return self.packtype(topic, packet.header, None)

class PictureInput(IO):
    def __init__(self, endpoint, topic, bind=True, hwm=None, multithreading=False):
        super().__init__(endpoint, topic, bind, hwm, socktype=zmq.SUB, packtype=Picture, multithreading=multithreading)

class U8PictureInput(PictureInput):
    def __init__(self, endpoint, topic, bind=True, hwm=None, multithreading=False):
        super().__init__(endpoint, topic, bind, hwm, multithreading=multithreading)
        self.packtype=U8Picture


class PictureOutput(IO):
    def __init__(self, endpoint, topic, bind=True, hwm=None, format=None, multithreading=False):
        super().__init__(endpoint, topic, bind, hwm, socktype=zmq.PUB, packtype=Picture, multithreading=multithreading)
        self.format = format
        if self.format is not None:
            validate_format(self.format)

    def response(self, packet):
        resp = super().response(packet)
        if self.format:
            resp.format = self.format
        return resp

class U8PictureOutput(PictureOutput):
    def __init__(self, endpoint, topic, bind=True, hwm=None, multithreading=False):
        super().__init__(endpoint, topic, bind, hwm, multithreading=multithreading)
        self.packtype=U8Picture

class JsonInput(IO):
    def __init__(self, endpoint, topic, bind=True, hwm=None, multithreading=False):
        super().__init__(endpoint, topic, bind, hwm, socktype=zmq.SUB, packtype=Json, multithreading=multithreading)

class JsonOutput(IO):
    def __init__(self, endpoint, topic, bind=True, hwm=None, multithreading=False):
        super().__init__(endpoint, topic, bind, hwm, socktype=zmq.PUB, packtype=Json, multithreading=multithreading)
