from ultralytics import YOLO 

model = YOLO('models/best.pt')  # Load a custom model

results = model.predict('input_videos/08fd33_4.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)