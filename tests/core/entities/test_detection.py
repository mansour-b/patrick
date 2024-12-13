import numpy as np

from patrick.core import Array, Frame, Model, NeuralNet


class TestModel:

    @staticmethod
    def make_model():
        class DumbModel(Model):
            def predict(self, frame: Frame) -> Frame:
                return frame

        return DumbModel()

    def test_init(self):
        _ = self.make_model()

    def test_predict(self):
        frame = Frame(name="frame_0", width=128, height=128, annotations=[])
        model = self.make_model()
        assert model.predict(frame) == frame


class TestNeuralNet:

    @staticmethod
    def make_net():
        class DumbNN(NeuralNet):
            def __call__(self, input_array: Array) -> Array:
                return input_array

        return DumbNN()

    def test_init(self):
        _ = self.make_net()

    def test_call(self):
        net = self.make_net()
        assert np.equal(net(np.zeros(3)), np.zeros(3)).max()
