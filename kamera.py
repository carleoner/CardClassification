from ultralytics import YOLO


model = YOLO("cards.pt")
model.info()
#source = n, gdzie n oznacza numer kamery w systemie:
#         0 = kamera z laptopa
#         1 = sztuczna kamera z OBS
#         2 = kamera z emulacji za pomocą DroidCam
results = model.predict(source="0", show=True)

print(results)

