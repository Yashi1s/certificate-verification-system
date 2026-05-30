import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from backend.extract_features import extract_features_from_image

dataset="dataset"

X=[]
y=[]

for label,folder in enumerate(["original","forged"]):

    path=os.path.join(dataset,folder)

    for file in os.listdir(path):

        img_path=os.path.join(path,file)

        features = extract_features_from_image(img_path)

        X.append(features)
        y.append(label)

X=np.array(X)
y=np.array(y)

model=RandomForestClassifier(n_estimators=200)

model.fit(X,y)

pickle.dump(model,open("backend/model.pkl","wb"))

print("Model trained successfully")