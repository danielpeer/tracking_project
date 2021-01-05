from tkinter import *

from processing_tracking.perform_tracking import perform_tracking

root = Tk()



background_image = PhotoImage(file = "D:\\Users\\97252\\PycharmProjects\\tracking_project\\GUI\\background.PNG")
background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

project_label = Label(root, text= "Welcome to tracking project", font = ("Helvetica",50))
project_label.pack(pady = 50)


def get_targets():
    value = input_file.get()
    perform_tracking(value)

input_file = Entry(root)
input_file.pack()
input_file.insert(0, "Please enter your video file path")

submit_input_file = Button(root, text = "Submit", command= get_targets)
submit_input_file.pack()


root.mainloop()