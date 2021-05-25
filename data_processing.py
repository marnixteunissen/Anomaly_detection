import os
import tensorflow.keras.preprocessing as preprocessing


def create_data_sets(data_dir, channel, mode, batch_size=32, image_size=[480, 360]):
    data_dir = os.path.join(data_dir, channel, mode)
    width, height = image_size[0], image_size[1]
    train_ds = preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(height, width),
        batch_size=batch_size)
    val_ds = preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(height, width),
        batch_size=batch_size)

    return train_ds, val_ds


def create_test_set(data_dir, channel, image_size=[480, 360], batch_size=1):
    data_dir = os.path.join(data_dir, channel, 'test')
    width, height = image_size[0], image_size[1]
    test_ds = preprocessing.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(height, width),
        batch_size=batch_size)

    return test_ds


if __name__ == "__main__":
    train, val = create_data_sets(r'data\data-set', 'TOP', 'train', batch_size=4)
    print('Classes:', train.class_names)
