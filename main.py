from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
import predict

app = FastAPI()
UPLOAD_DIR ="uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)



@app.get("/")
def read_root():
    return {"Hello" : "World"}



@app.post("/car-image/upload/")
async def uplaod_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, 'wb') as buffer:  
        shutil.copyfileobj(file.file, buffer)

    image_path = f"uploads/{file.filename}"
    car_details = predict.read_lp_text(image_path)
    
   
    
    return {"car-details " : car_details}

