import unittest
from fruit_classifier.models.models import get_lenet


class TestModels(unittest.TestCase):
    def test_get_lenet(self):
        intended_height = 7
        intended_width = 8
        intended_channels = 5
        intended_classes = 4

        model = get_lenet(intended_height, intended_width,
                          intended_channels, intended_classes)
        shape = model.input_shape
        model_height = shape[1]
        model_width = shape[2]
        model_channels = shape[3]

        self.assertEqual(intended_height, model_height)
        self.assertEqual(intended_width, model_width)
        self.assertEqual(intended_channels, model_channels)

        num_layers = len(model.layers)
        final_layer = model.layers[num_layers-1]
        num_classes = final_layer.output_shape[1]
        self.assertEqual(intended_classes, num_classes)


if __name__ == '__main__':
    unittest.main()
