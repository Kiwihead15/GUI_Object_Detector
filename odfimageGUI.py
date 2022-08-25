# %% [markdown]
# # Image Detector from picture

# %% [markdown]
# This script detect objects from a picture. 
# ML model is yolo4.  It was trained with 320x320 images to detect the following objects:
# 
#  ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# 
# 
# 

# %% [markdown]
# ## Import modules

# %%
import cv2
from tkinter import *     # from tkinter import Tk for Python 3.x
from tkinter import mainloop
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from numpy import append

# %%
# parameters of the model
net = cv2.dnn.readNet('yolov4-tiny.weights','yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(net)
size = (320,320)
model.setInputParams(size=size, scale=1/255)

# %%
root = Tk()
root.geometry("280x800")
root.title("Object Detector")
root.iconbitmap('Object-Detector-120.png')

# Read the objects to be detected. Build a list.
classes=[]
t = Text(root,width=15, height=20)
objects= ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

clicked = StringVar()
clicked.set(objects[0])

# Drop menu to select objects to be detected by AI
drop = OptionMenu(root, clicked, *objects)
drop.grid(row=4, column=0)


def show():
    global classes
    global objects
    #class_name = Label(root, text=clicked.get())
    #class_name.grid(row=5, column=0, columnspan=2)
    object_chosen = clicked.get()
    if object_chosen in classes:
        response = messagebox.showinfo('Popup Info','Object already selected')
        Label(root, text=response).pack()
    else:
        classes.append(object_chosen)
        t.insert(END, object_chosen + '\n')
        t.grid(row=6, column=0, columnspan=2)
    status_label=Label(root, text='Object added to the list')
    status_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
    start_btn.configure(state='normal')

#Button to add object to a list detection
Add_object = Button(root, text="Add object to the list", command=show)
Add_object.grid(row=4, column=1)

# Select an image function
def open_image():
    global frame
    global status_label 

    path_img = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    img = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)
    scale_percent = size[0]/img.shape[0] #percent by which the image is resized
    dsize = ( int(img.shape[1]*scale_percent), size[0]) # dsize
    frame = cv2.resize(img, dsize) # resize image
    cv2.imshow("Frame", frame)
    status_label=Label(root, text='Image Selected, proceed with objects selection')
    status_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
    #start_btn.configure(state='normal')
    return 

def start_detection():
    global frame
    global status_label 
    
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=0.4) #model detection
    cv2.destroyAllWindows()
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        class_id = int(class_id)    # need just for cv2 4.5.3
        (x, y, w , h) = bbox
        class_name = objects[class_id]
        #color = colors[class_id] 

        if class_name in classes:
            cv2.putText(frame,str(class_name), (x,y-10), cv2.FONT_HERSHEY_PLAIN,2, 2)  #color
            cv2.rectangle(frame,(x,y),(x+w,y+h),3) #color

    cv2.imshow("Frame",frame)     
    status_label=Label(root, text='                  Starting Detection                  ')
    status_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
    stop_btn.configure(state='normal')
    save_btn.configure(state='normal')
    open_btn.configure(state='disable')
    return


def stop_detection():
    global status_label 

    cv2.destroyAllWindows()
    status_label=Label(root, text='                  Detection Stopped                  ')
    status_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
    save_btn.configure(state='normal')
    start_btn.configure(state='normal')
    open_btn.configure(state='normal')
    return 

def save_image():
    global frame
    global status_label 
    image_file = asksaveasfilename( defaultextension=".jpg", filetypes=(("jpg files", "*.jpg"),("All Files", "*.*")),)
    if image_file:
        cv2.imwrite(image_file,frame)
    status_label=Label(root, text='                  Image has been saved                  ')
    status_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
    return

def close():
    cv2.destroyAllWindows()
    root.destroy()


#Buttons for file
open_btn=Button(root, text = "Select Image", bg = "green", command=open_image, width=15)
open_btn.grid(row = 1, column = 0, padx=10, pady=10)
save_btn=Button(root, text = "Save Image", bg = "red",command=save_image, width=15, state = 'disable')
save_btn.grid(row=1, column = 1, padx=10, pady=10)

#Buttons for detection
start_btn=Button(root, text = "Start Detection", bg = "green", command = start_detection, width=15, state = 'disable')
start_btn.grid(row = 2, column = 0, padx=10, pady=10)
stop_btn=Button(root, text = "Stop Detection", bg = "red", command = stop_detection, width=15, state = 'disable')
stop_btn.grid(row = 2, column = 1, padx=10, pady=10)

#Button for closing the app.
close_btn = Button(root, text="EXIT PROGRAM", bg = "red", command=close)
close_btn.grid(row = 7, column = 1, padx=10, pady=10)

status_label=Label(root, text='Select an image where to detect objects')
status_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()


