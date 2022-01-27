import onnxruntime as ort
import os
import argparse
import cv2


if __name__ == "__main__":

    session = ort.InferenceSession("models/onnx/version-RFB-320.onnx")
    input_name = session.get_inputs()[0].name
    print(input_name)
