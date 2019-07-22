import cv2
import numpy as np


class DataFeeder:
    def __init__(self, images=None, labels=None, batch_size=1):
        self.images = images
        self.labels = labels
        self.num_classes = np.max(labels) + 1
        self.batch_size = batch_size

    def set_images(self, images):
        self.images = images

    def set_labels(self, labels):
        self.labels = labels

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    @staticmethod
    def brightness_augment(img, factor=0.2):
        img_hsv = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), dtype=np.float64)
        img_hsv[:, :, 2] = img_hsv[:, :, 2] * np.random.uniform(1 - factor, 1 + factor)
        img_hsv[:, :, 2][img_hsv[:, :, 2] > 255] = 255
        img_rgb = cv2.cvtColor(np.array(img_hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
        return img_rgb

    @staticmethod
    def flip_augment(img, factor=0.5):
        if np.random.random() > factor:
            return np.fliplr(img)
        else:
            return img

    @staticmethod
    def add_noise(img, factor=0.15):
        width, height, channel = img.shape
        noise_arr = (np.random.rand(width, height, channel) - 0.5) * factor * 255.0
        img_noise = img + noise_arr
        return np.minimum(np.maximum(img_noise, 0.0), 255.0)

    @staticmethod
    def crop_augment(img, factor=0.5, limit=3):
        if np.random.random() > factor:
            shape_orig = img.shape
            x_left = np.random.randint(0, limit)
            x_right = np.random.randint(shape_orig[0] - limit, shape_orig[0])
            y_top = np.random.randint(0, limit)
            y_bottom = np.random.randint(shape_orig[1] - limit, shape_orig[1])
            img_crop = img[x_left:x_right, y_top:y_bottom, :]
            img_crop_t = cv2.resize(img_crop, dsize=shape_orig[0:2],interpolation=cv2.INTER_LINEAR)
            return img_crop_t
        else:
            return img

    def next_batch(self):
        batch_index = np.random.choice(np.arange(self.labels.size), size=self.batch_size, replace=False)
        labels_batch = self.labels[batch_index]
        data_batch_orig = self.images[batch_index]
        data_batch_ls = []
        for i in range(self.batch_size):
            img_aug = self.brightness_augment(data_batch_orig[i])
            img_aug = self.add_noise(img_aug)
            if labels_batch[i] in [9, 11, 12, 13, 15, 17, 18, 22, 26, 29, 30, 35]:
                img_aug = self.flip_augment(img_aug)
            img_aug = self.crop_augment(img_aug)
            data_batch_ls.append(img_aug)
        data_batch = np.stack(data_batch_ls)
        return data_batch, labels_batch


if __name__ == '__main__':
    import data_loader

    x_train, y_train = data_loader.x_train, data_loader.y_train
    for i in range(43):
        y_sub = y_train[y_train == i]
        x_sub = x_train[y_train == i]
        sub_index = np.random.choice(np.arange(y_sub.size), size=2, replace=False)
        x_t = x_sub[sub_index]
        for j in range(len(x_t)):
            img_t = np.squeeze(x_t[j])
            img_r = cv2.cvtColor(img_t, cv2.COLOR_RGB2BGR)
            name = "pic/%02d_%02d.jpg" % (i, j)
            cv2.imwrite(name, img_r)

    data_feeder = DataFeeder(x_train, y_train, batch_size=4)

    data_batch, labels_batch = data_feeder.next_batch()
    print()
