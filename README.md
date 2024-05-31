<a name="48ea60a3"></a>
# 1 实验内容
- 进一步掌握TensorFlow模型训练和生成的基本流程
- 下载石头剪刀布图片数据集
- 学习石头剪刀布图片识别模型的生成，参考教程。
- 绘制图像验证模型的性能
<a name="iSUqf"></a>
# 2 实验记录
<a name="wsDom"></a>
## 2.1 准备过程
<a name="kUame"></a>
### 2.1.1 下载数据集
在终端内输入命令，将石头剪刀布的训练集和测试集下载至当前目录：<br />`wget --no-check-certificate https://storage.googleapis.com/learning-datasets/rps.zip -O ./rps.zip`<br />`wget --no-check-certificate https://storage.googleapis.com/learning-datasets/rps-test-set.zip -O ./rps-test-set.zip`<br />![9YZF}AT8KK@`C9FI`_Y5C]B.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717157013548-bf97d575-0e38-4a3b-a393-1df034fc8889.png#averageHue=%23f4f2f1&clientId=u3ca7d73f-f97c-4&from=paste&height=192&id=u258581fc&originHeight=264&originWidth=1473&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=12931&status=done&style=none&taskId=uddfc0e9b-cb3c-4130-9acb-c793d00e922&title=&width=1071.2727272727273)<br />![UOPY[_T@SC_S4WKW(C$@$Z3.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717157027240-81569feb-0e97-4b4f-9cb8-9ec199e93e73.png#averageHue=%23f4f2f1&clientId=u3ca7d73f-f97c-4&from=paste&height=196&id=u5c10cd23&originHeight=269&originWidth=1502&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=15275&status=done&style=none&taskId=ub131eb1d-f3ec-44ec-9f0c-f42dbe05841&title=&width=1092.3636363636363)<br />![QU[1M$77L0{GKR[R$7F2VV8.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717156961695-c33d9025-ec01-469d-8a9b-3a6a1e48a6fb.png#averageHue=%23f5e7e5&clientId=u3ca7d73f-f97c-4&from=paste&height=155&id=uf7897e68&originHeight=213&originWidth=344&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=11789&status=done&style=none&taskId=udaf8c14e-1296-47f5-9f47-5fd534ce60c&title=&width=250.1818181818182)<br />通过下面代码解压数据集到当前目录：
```python
import os
import zipfile

local_zip = './rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./')
zip_ref.close()

local_zip = './rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./')
zip_ref.close()
```
![_$5XSXRR1~DX3U5_9%0[QI8.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717157079881-4e2b366b-cf19-41c5-b31e-aa91eb76ec29.png#averageHue=%23f6f6f5&clientId=u3ca7d73f-f97c-4&from=paste&height=81&id=ubf516447&originHeight=88&originWidth=337&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=1253&status=done&style=none&taskId=uc3031296-c911-497b-a6d3-33a65372d96&title=&width=309.0909118652344)
<a name="b7111c90"></a>
### 2.1.2 验证数据集
打印数据集的相关信息以验证数据集的完整性。
```python
rock_dir = os.path.join('./rps/rock')
paper_dir = os.path.join('./rps/paper')
scissors_dir = os.path.join('./rps/scissors')

# 各手势数据集包含的图片张数
print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

# 输出各数据集的前5张图片的文件名
rock_file = os.listdir(rock_dir)
rock_file.sort()
print(rock_file[:5])
paper_file = os.listdir(paper_dir)
paper_file.sort()
print(paper_file[:5])
scissors_file = os.listdir(scissors_dir)
scissors_file.sort()
print(scissors_file[:5])
```

```
total training rock images: 840
total training paper images: 840
total training scissors images: 840
['rock01-000.png', 'rock01-001.png', 'rock01-002.png', 'rock01-003.png', 'rock01-004.png']
['paper01-000.png', 'paper01-001.png', 'paper01-002.png', 'paper01-003.png', 'paper01-004.png']
['scissors01-000.png', 'scissors01-001.png', 'scissors01-002.png', 'scissors01-003.png', 'scissors01-004.png']
```

从石头、剪刀、布训练集中分别打印两张图片：
```python
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 100

next_rock = [os.path.join(rock_dir, fname)
         for fname in rock_file[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname)
         for fname in paper_file[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname)
         for fname in scissors_file[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock + next_paper + next_scissors):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()
```
![](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717157923831-3c8d0826-2b4e-4282-9278-bfbcb1c57c70.png#averageHue=%23f4efee&from=url&id=ONavL&originHeight=389&originWidth=389&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)![](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717157923910-6030ba32-58c3-4941-9181-caa4c7be9407.png#averageHue=%23f5efee&from=url&id=roOHN&originHeight=389&originWidth=389&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)<br />![](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717157923982-ba57988e-9d6e-43eb-8d8f-f979f78cc75d.png#averageHue=%23e4dddc&from=url&id=JCdNb&originHeight=389&originWidth=389&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)<br />![](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717157924063-ab859f65-8f94-4837-a3d9-12b1edfce7b9.png#averageHue=%23e4dddc&from=url&id=G1IKY&originHeight=389&originWidth=389&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)<br />![](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717157924127-86d5ca50-1c1d-4e30-b3a9-655b8fdd1a87.png#averageHue=%23f2edeb&from=url&id=Aq10q&originHeight=389&originWidth=389&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)<br />![](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717157924194-116e3d02-e15b-4188-8fe0-81c53afcd88e.png#averageHue=%23f2edeb&from=url&id=duiMH&originHeight=389&originWidth=389&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
<a name="962c3878"></a>
## 2.2 模型生成
调用TensorFlow的keras进行数据模型的训练，ImageDataGenerator是Keras中图像预处理的类，经过预处理使得后续的训练更加准确。
```python
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = './rps/'
# 图片预处理
training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

VALIDATION_DIR = "./rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=126
)

# 定义模型架构
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

# 编译模型并定义相关参数
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

# 用训练数据拟合
history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)
```

```
2024-05-31 11:19:08.814399: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2024-05-31 11:19:08.814449: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.


Found 2520 images belonging to 3 classes.
Found 372 images belonging to 3 classes.


2024-05-31 11:19:20.168223: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2024-05-31 11:19:20.168268: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2024-05-31 11:19:20.168292: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (codespaces-4bb514): /proc/driver/nvidia/version does not exist
2024-05-31 11:19:20.168662: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 64)      1792      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 74, 74, 64)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 72, 72, 64)        36928     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 17, 17, 128)      0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 15, 15, 128)       147584    
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 7, 7, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 6272)              0         
                                                                 
 dropout (Dropout)           (None, 6272)              0         
                                                                 
 dense (Dense)               (None, 512)               3211776   
                                                                 
 dense_1 (Dense)             (None, 3)                 1539      
                                                                 
=================================================================
Total params: 3,473,475
Trainable params: 3,473,475
Non-trainable params: 0
_________________________________________________________________


2024-05-31 11:19:21.801546: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 34020000 exceeds 10% of free system memory.


Epoch 1/25


2024-05-31 11:19:25.190819: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 34020000 exceeds 10% of free system memory.
2024-05-31 11:19:25.210296: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 706535424 exceeds 10% of free system memory.
2024-05-31 11:19:26.339031: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 176633856 exceeds 10% of free system memory.
2024-05-31 11:19:26.516623: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 167215104 exceeds 10% of free system memory.


20/20 [==============================] - 131s 6s/step - loss: 1.2608 - accuracy: 0.3706 - val_loss: 1.0801 - val_accuracy: 0.3333
Epoch 2/25
20/20 [==============================] - 118s 6s/step - loss: 1.1010 - accuracy: 0.3849 - val_loss: 1.0713 - val_accuracy: 0.3333
Epoch 3/25
20/20 [==============================] - 124s 6s/step - loss: 1.0885 - accuracy: 0.4710 - val_loss: 0.7870 - val_accuracy: 0.6532
Epoch 4/25
20/20 [==============================] - 118s 6s/step - loss: 0.9760 - accuracy: 0.5397 - val_loss: 0.8186 - val_accuracy: 0.6183
Epoch 5/25
20/20 [==============================] - 117s 6s/step - loss: 0.8621 - accuracy: 0.6306 - val_loss: 0.4621 - val_accuracy: 0.9462
Epoch 6/25
20/20 [==============================] - 119s 6s/step - loss: 0.6611 - accuracy: 0.6937 - val_loss: 0.8699 - val_accuracy: 0.6075
Epoch 7/25
20/20 [==============================] - 120s 6s/step - loss: 0.5948 - accuracy: 0.7313 - val_loss: 0.3731 - val_accuracy: 0.9704
Epoch 8/25
20/20 [==============================] - 117s 6s/step - loss: 0.5071 - accuracy: 0.8024 - val_loss: 0.4726 - val_accuracy: 0.7339
Epoch 9/25
20/20 [==============================] - 118s 6s/step - loss: 0.4129 - accuracy: 0.8313 - val_loss: 0.0822 - val_accuracy: 0.9919
Epoch 10/25
20/20 [==============================] - 119s 6s/step - loss: 0.3248 - accuracy: 0.8671 - val_loss: 0.0924 - val_accuracy: 0.9866
Epoch 11/25
20/20 [==============================] - 119s 6s/step - loss: 0.2439 - accuracy: 0.9044 - val_loss: 0.0476 - val_accuracy: 0.9866
Epoch 12/25
20/20 [==============================] - 118s 6s/step - loss: 0.2323 - accuracy: 0.9123 - val_loss: 0.0428 - val_accuracy: 0.9919
Epoch 13/25
20/20 [==============================] - 119s 6s/step - loss: 0.3379 - accuracy: 0.8861 - val_loss: 0.5081 - val_accuracy: 0.7554
Epoch 14/25
20/20 [==============================] - 117s 6s/step - loss: 0.1513 - accuracy: 0.9488 - val_loss: 0.0623 - val_accuracy: 0.9839
Epoch 15/25
20/20 [==============================] - 118s 6s/step - loss: 0.1596 - accuracy: 0.9433 - val_loss: 0.0298 - val_accuracy: 1.0000
Epoch 16/25
20/20 [==============================] - 120s 6s/step - loss: 0.2229 - accuracy: 0.9210 - val_loss: 0.3965 - val_accuracy: 0.8145
Epoch 17/25
20/20 [==============================] - 118s 6s/step - loss: 0.1388 - accuracy: 0.9516 - val_loss: 0.1016 - val_accuracy: 0.9677
Epoch 18/25
20/20 [==============================] - 118s 6s/step - loss: 0.0923 - accuracy: 0.9671 - val_loss: 0.0179 - val_accuracy: 1.0000
Epoch 19/25
20/20 [==============================] - 120s 6s/step - loss: 0.1217 - accuracy: 0.9603 - val_loss: 0.1435 - val_accuracy: 0.9462
Epoch 20/25
20/20 [==============================] - 118s 6s/step - loss: 0.2032 - accuracy: 0.9306 - val_loss: 0.0654 - val_accuracy: 0.9731
Epoch 21/25
20/20 [==============================] - 117s 6s/step - loss: 0.0822 - accuracy: 0.9746 - val_loss: 0.0143 - val_accuracy: 1.0000
Epoch 22/25
20/20 [==============================] - 117s 6s/step - loss: 0.0809 - accuracy: 0.9738 - val_loss: 0.3009 - val_accuracy: 0.8280
Epoch 23/25
20/20 [==============================] - 117s 6s/step - loss: 0.1456 - accuracy: 0.9484 - val_loss: 0.0570 - val_accuracy: 0.9677
Epoch 24/25
20/20 [==============================] - 118s 6s/step - loss: 0.0575 - accuracy: 0.9825 - val_loss: 0.3360 - val_accuracy: 0.8495
Epoch 25/25
20/20 [==============================] - 122s 6s/step - loss: 0.1039 - accuracy: 0.9635 - val_loss: 0.0419 - val_accuracy: 0.9785
```
将生成的模型以h5格式保存到当前目录
```python
model.save("rps.h5")
```
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717157425173-0073bd07-405c-48dc-a0a2-f0fc1160b512.png#averageHue=%23f4e9e8&clientId=u3ca7d73f-f97c-4&from=paste&height=153&id=ud8683a26&originHeight=210&originWidth=386&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=9227&status=done&style=none&taskId=u8996e79d-bc21-4d88-b7c8-97b999486b3&title=&width=280.72727272727275)
<a name="GRpo7"></a>
# 3 模型评估
通过绘制图形的方式，验证生成模型的性能
```python
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
```
![](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717157924275-de09c52f-36bc-4a83-807b-4c7970590165.png#averageHue=%23faf8f8&from=url&id=r0iyK&originHeight=435&originWidth=552&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

```
<Figure size 432x288 with 0 Axes>
```
