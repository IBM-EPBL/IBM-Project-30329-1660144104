import numpy as np
import os
import pandas as pd
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request,url_for

app=Flask(__name__)
model=load_model("vegetable.h5")
model1=load_model("fruit.h5")
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/prediction')
def prediction():
    return render_template("predict.html")

@app.route('/prediction1',methods=['GET','POST'])
def predict_img():
    f=request.files['image']
    basepath=os.path.dirname(__file__)
    filepath=os.path.join(basepath,'uploads',secure_filename(f.filename))
    f.save(filepath)
    img=image.load_img(filepath, target_size=(64,64))
    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    plant=request.form['plant']
    print(plant)
    if(plant=="vegetable"):
           preds = model.predict(x)
           preds =np.argmax(preds)
           print(preds)
           df=pd.read_excel('precautions - veg.xlsx')
           print(df.iloc[preds]['caution'])
    else:
            preds = model1.predict(x)
            preds =np.argmax(preds)     
            df=pd.read_excel('precautions - fruits.xlsx')
            print(df.iloc[preds]['caution'])
            
        
    return (df.iloc[preds]['caution'])
    
    
if __name__=="__main__":
 app.run(debug=False)            