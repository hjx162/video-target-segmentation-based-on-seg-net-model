# -*- coding: UTF-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import numpy as np


print(os.listdir("E:/Python/pythonProject_4/target_tracking_and_detection/tmp/archive/images/")[-5:])
img = tf.io.read_file("E:/Python/pythonProject_4/target_tracking_and_detection/tmp/archive/images/yorkshire_terrier_99.jpg")
img = tf.image.decode_png(img)
img = tf.squeeze(img)
plt.imshow(img)
plt.show()

img1 = tf.io.read_file("E:/Python/pythonProject_4/target_tracking_and_detection/tmp/annotations/trimaps/yorkshire_terrier_99.png")
img1 = tf.image.decode_png(img1)
plt.imshow(img1)
plt.show()

#读取所有的图片
images = glob.glob("E:/Python/pythonProject_4/target_tracking_and_detection/tmp/archive/images/*.jpg")
# glob.glob('*g') : 匹配所有的符合条件/格式的文件，并将其以list的形式返回
print(len(images))
anno = glob.glob("E:/Python/pythonProject_4/target_tracking_and_detection/tmp/annotations/trimaps/*.png")
print(len(anno))

images.sort(key=lambda x:x.split('\\')[1].split('.')[0])
anno.sort(key=lambda x:x.split('\\')[1].split('.')[0])

# 打乱
np.random.seed(2021)
index = np.random.permutation(len(images))
# np.random.permutation() : 返回一个乱序的序列
images = np.array(images)[index]
anno = np.array(anno)[index] # 打乱，且list格式转换成array格式

# 读取原图像路径
def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img,channels=3)
    return img

# 读取分割图像路径
def read_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img,channels=1)
    return img

# 图像归一化函数  归一化范围 ：[-1, 1]
def normal_img(input_images,input_anno):
    input_images = tf.cast(input_images,tf.float32) # 数据类型转换 : 将读入图片的int8类型转换为float32类型
    input_images = input_images/127.5 - 1
    input_anno = input_anno -1
    return input_images,input_anno

# 图像载入
def load_image(input_images_path,input_anno_path):
    input_images = read_jpg(input_images_path)
    input_anno = read_png(input_anno_path)
    input_images = tf.image.resize(input_images,(224,224))
    input_anno = tf.image.resize(input_anno,(224,224))
    return normal_img(input_images,input_anno)

AUTOTUNE = tf.data.experimental.AUTOTUNE # 设置数据预取缓冲的元素数量为自动默认
dataset = tf.data.Dataset.from_tensor_slices((images,anno)) # 将目标图片地址和分割标注地址一一对应再分离(实质是把数据合并降到0维，每个0维内存在目标图片和其对印的标注图片)
dataset = dataset.map(load_image,num_parallel_calls=AUTOTUNE) # 设置数据处理形式：并行处理

# 设置训练数据和验证集数据的大小
test_count = int(len(images)*0.2)
train_count = len(images) - test_count
print(test_count,train_count)
# 跳过test_count个
train_dataset = dataset.skip(test_count) # 个人理解为从dataset数据集中选出test_count个对应数据以外的数据
test_dataset = dataset.take(test_count) # 个人理解为从dataset数据集中选出test_count个对应的数据

batch_size = 8
# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据被充分打乱。
train_ds = train_dataset.shuffle(buffer_size=train_count).repeat().batch(batch_size) # batch_size:取样间隔 ; repeat:epoch次数
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # 设置数据预取缓冲的元素数量为自动默认
test_ds = test_dataset.batch(batch_size)
test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

for image,anno in train_ds.take(1):
    plt.subplot(1,2,1)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image[0]))
    plt.subplot(1,2,2)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(anno[0]))
    plt.show()

vgg16 = tf.keras.applications.VGG16(input_shape=(224, 224, 3),
                          include_top=False,
                          weights='imagenet')
vgg16.summary()

# python中super().__init__()：https://blog.csdn.net/a__int__/article/details/104600972
class Connect(tf.keras.layers.Layer): # 继承类：tf.keras.layers.Layer
    def __init__(self,
                 filters=256,
                 name='Connect',
                 **kwargs): # *args和**kw分别属于非关键字参数和关键字参数，也都是可变参数。
        super(Connect, self).__init__(name=name, **kwargs)

        self.Conv_Transpose = tf.keras.layers.Convolution2DTranspose(filters=filters,
                                                                     kernel_size=3,
                                                                     strides=2,
                                                                     padding="same",
                                                                     activation="relu")

        self.conv_out = tf.keras.layers.Conv2D(filters=filters,
                                               kernel_size=3,
                                               padding="same",
                                               activation="relu")

    def get_config(self):
        config = ({
            'Conv_Transpose': self.Conv_Transpose,
            'conv_out': self.conv_out,
        })
        base_config = super(Connect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs): # call()的本质是将一个类变成一个函数（使这个类的实例可以像函数一样调用）。
        x = self.Conv_Transpose(inputs)
        return self.conv_out(x)


layer_names = ["block5_conv3",
               "block4_conv3",
               "block3_conv3",
               "block5_pool"]
# 得到4个输出
layers_out = [vgg16.get_layer(layer_name).output for layer_name in layer_names]
multi_out_model = tf.keras.models.Model(inputs=vgg16.input,
                                        outputs=layers_out)
multi_out_model.trainable = False #

# 创建输入
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
out_block5_conv3, out_block4_conv3, out_block3_conv3, out = multi_out_model(inputs)
print(out_block5_conv3.shape)

x1 = Connect(512, name="connect_1")(out)
x1 = tf.add(x1, out_block5_conv3)  # 元素对应相加,加入编码器中对应的特征映射

x2 = Connect(512, name="connect_2")(x1)
x2 = tf.add(x2, out_block4_conv3)  # 元素对应相加

x3 = Connect(256, name="connect_3")(x2)
x3 = tf.add(x3, out_block3_conv3)  # 元素对应相加

x4 = Connect(128, name="connect_4")(x3)

prediction = tf.keras.layers.Convolution2DTranspose(filters=3,
                                                    kernel_size=3,
                                                    strides=2,
                                                    padding="same",
                                                    activation="softmax")(x4)

model = tf.keras.models.Model(inputs=inputs, outputs=prediction)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"]
       )

steps_per_eooch = train_count//batch_size
validation_steps = test_count//batch_size

history = model.fit(train_ds,
            epochs=3,
            steps_per_epoch=steps_per_eooch,
            validation_data=test_ds,
            validation_steps=validation_steps)


tf.keras.models.save_model(model,'E:/Python/pythonProject_4/target_tracking_and_detection/model.h5') # model 保存