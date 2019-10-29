import cv2
import pickle
from keras.models import load_model

image_path = r'C:\Users\Serban\Desktop\crack-detector\CommandLineApp\TestImages\09513.jpg'
model_path = r'C:\Users\Serban\Desktop\crack-detector\CommandLineApp\Model\crack.model'
label_path = r'C:\Users\Serban\Desktop\crack-detector\CommandLineApp\Model\crack.pickle'

image = cv2.imread(image_path)

output_img = image.copy()

image = cv2.resize(image, (32, 32))

image = image.astype(float) / 255.0

image = image.reshape((1, image.shape[0] * image.shape[1] * image.shape[2]))

print("[INFO] loading network and label binarizer...")
model = load_model(model_path)
lb = pickle.loads(open(label_path, "rb").read())

pred = model.predict(image)

i = pred.argmax(axis=1)[0]
label = lb.classes_[i]

text = "{}: {:.2f}%".format(label, pred[0][i] * 100)
cv2.putText(output_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow("Image", output_img)
cv2.waitKey(0)
