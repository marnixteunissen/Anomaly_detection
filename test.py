import tensorflow as tf
from data_processing import create_test_set
from models import build_conv_network
import data_processing
import os


def evaluate_network(model, data_dir, channel='TOP'):
    test_set = create_test_set(data_dir, channel)
    result = model.evaluate(test_set)
    dict(zip(model.metric_names, result))


if __name__ == "__main__":
    data_dir = os.getcwd() + r'\data\data-set'
    train_set, val_set = data_processing.create_data_sets(data_dir, 'TOP', 'train')

    test_model = build_conv_network(3, 16)
    test_model.fit(train_set, validation_data=val_set, epochs=3)

    evaluate_network(test_model, data_dir)
