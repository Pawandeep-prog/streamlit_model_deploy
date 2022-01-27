import streamlit as st
from keras.models import load_model 
import numpy as np 

model = load_model("model.h5")
labels = np.load("labels.npy") 

st.title("Welcome to flower prediction app")

a = float(st.number_input("sepal length in cm"))
b = float(st.number_input("sepal width in cm"))
c = float(st.number_input("petal length in cm"))
d = float(st.number_input("petal width in cm"))

btn = st.button("predict")

if btn:
	pred = model.predict(np.array([a,b,c,d]).reshape(1,-1))
	pred = labels[np.argmax(pred)]
	st.subheader(pred)

	if pred=="Iris Setosa":
		st.image("setosa.jpg")
	elif pred=="Iris Versicolour":
		st.image("versicolor.jpg")
	else:	
		st.image("verginca.jpg")

