import cv2
import numpy as np
from abc import ABC, abstractmethod

class FrameSource(ABC):

    @abstractmethod
    def read():
         pass
    
    @abstractmethod
    def release():
         pass