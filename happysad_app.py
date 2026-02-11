import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. ตั้งชื่อหน้าเว็บ
st.title("😊 Happy vs ☹️ Sad Classifier")
st.write("อัปโหลดรูปภาพเพื่อให้ AI ทายความรู้สึกกัน!")

# 2. โหลดโมเดล (ใส่ @st.cache_resource เพื่อให้โหลดแค่ครั้งเดียว ไม่เปลือง RAM)
@st.cache_resource
def load_my_model():
    return load_model('imageclassifier_pretrained_happysad.keras')

model = load_my_model()

# 3. ส่วนการรับรูปภาพจากผู้ใช้
uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงรูปที่อัปโหลด
    img = Image.open(uploaded_file)
    st.image(img, caption='รูปที่อัปโหลด', use_column_width=True)
    
    # 4. เตรียมรูปให้พร้อมสำหรับ AI (Preprocess)
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # เพิ่มมิติ Batch

    # 5. สั่ง AI ทายผล
    if st.button('ทำนายผล!'):
        prediction = model.predict(img_array)
        
        if prediction > 0.5:
            st.error(f"ผลลัพธ์คือ: SAD (มั่นใจ {prediction[0][0]*100:.2f}%)")
        else:
            st.success(f"ผลลัพธ์คือ: HAPPY (มั่นใจ {(1-prediction[0][0])*100:.2f}%)")