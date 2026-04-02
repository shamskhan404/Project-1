# Real-Time Emotion Detection (RTED)

This app uses a deep learning model to classify emotions from images and real-time video streams.

###DEMO 
https://huggingface.co/spaces/ShamsKhan404/Real_Time_Emotion_Detection

### How It Works
1. **Choose a Detection Mode:**  
   - **Static Detection:** Upload an image.  
   - **Real-Time Detection:** (Currently disabled due to Hugging Face webcam restrictions).  

2. **Model Predictions:**  
   - The model predicts emotions from 7 categories:  
     **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise**  

3. **File Structure:**  
   - **Model File:** `model/RAFDB_Custom.h5` (Ensure this exists)  
   - **Static Files:** Inside `static/`  

---

### Running the App Locally  
```bash
pip install -r requirements.txt
python app.py
