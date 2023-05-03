label=['0','1','2'] #how many ever classes in dataset
def convert(path): 
    img_array = cv2.imread(path)
    new_array = cv2.resize(img_array, (100, 100))
    return new_array.reshape(-1, 50, 50, 3)
    
def print_image(path, title): 
    image = cv2.imread(path)
    plt.figure(figsize=(3, 3))
    plt.axis("off")
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def predict_class(im_path): 
    img = convert(im_path)
    prediction = model.predict(img)
    pred_label = label[prediction.argmax()]
    print_image(im_path, pred_label)


predict_class('[path of image]')
