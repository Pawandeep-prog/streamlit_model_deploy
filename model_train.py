from keras.models import Model 
from keras.layers import Input, Dense  
import pandas as pd 
from tensorflow.keras.utils import to_categorical
import numpy as np

data = pd.read_csv('iris.data').values

rand = np.arange(149)
np.random.shuffle(rand)
c = 0

copy = data.copy()

for i in rand:
	data[c] = copy[i]
	c = c+1

print(data)

X = data[:, :4]
y = data[:, 4]
print(y)
labels = {}
cnt = 0 

for i in y:
	if i not in labels:
		labels[i] = cnt  
		cnt = cnt + 1 

print("="*50)
print(labels)
for i in range(y.shape[0]):
	y[i] = labels[y[i]]
print(y)
y = to_categorical(y)
print(y)

X= np.array(X, dtype="float64")
y= np.array(y, dtype="float64")

print(X.shape, y.shape)

inp = Input(shape=(4))

x = Dense(32, activation="relu")(inp)

op = Dense(3, activation="softmax")(x)

model=  Model(inputs=inp, outputs=op)

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['acc'])

model.fit(X, y, epochs=30)

model.save('model.h5')

arr = []

for k in labels.keys():
	arr.append(k)
	
print(arr)
print(labels)
np.save("labels.npy", np.array(arr))