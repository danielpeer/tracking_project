from tkinter import Tk, Label, PhotoImage, Entry, Button, Canvas
from tkinter.filedialog import askopenfilename, Misc, asksaveasfile, asksaveasfilename
from PIL import ImageTk, Image
import cv2
from processing_tracking.perform_tracking import perform_tracking
import os

males = 0
female = 0

play_original_video = 1
play_processed_video = 2
general = 3


output_video = "D:\\Users\\97252\PycharmProjects\\tracking_project\\GUI\\berlin_walk.avi"


class App:
    def __init__(self):
        self.root = self.define_root()
        self.canvas = Canvas(self.root, width=1200, height=800)
        self.background_image = None
        self.current_frame = 0
        self.gender_dict = {'male': [], 'female': [], "outgoing_targets": [], "current_targets": []}
        self.is_should_init = True
        self.cap = None
        self.lmain = None
        self.input_video = None
        self.state = general
        self.First_Screen()
        self.root.mainloop()

    def define_root(self):
        root = Tk()
        root.title("tracking project")
        root.geometry("1200x800")
        return root

    def First_Screen(self):
        self.background_image = PhotoImage(
            file= ".\\background.PNG")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.background_image, anchor="nw")
        self.canvas.create_text(700, 250, text="Welcome to tracking project", font=("Helvetica", 50), fill='white')
        file_btn = Button(self.root, text="Open file", command=self.open_video)
        self.canvas.create_window(750, 400, window=file_btn)
        self.is_should_init = True
        self.root.update()

    def process_video(self, filename):
        self.background_image = PhotoImage(file=".\\network.png")
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.background_image, anchor="nw")
        self.canvas.create_text(700, 250, text="Processing the video, please wait ...", font=("Helvetica", 50),
                                fill='white')
        self.root.update()
        perform_tracking(filename, self.gender_dict)
        self.video_screen(None)

    def clear_labels(self):
        for label in self.root.children.values(): label.destroy()

    def open_video(self):
        filename = askopenfilename(title="Please select your video",
                                   filetypes=(("avi files", "*.avi"), ("mp4 files", "*.mp4")))
        self.input_video = filename
        self.process_video(filename)

    def return_to_First_Screen(self):
        self.state = general
        self.is_should_init = True
        self.canvas.delete("all")
        self.First_Screen()

    def save_as_file(self):
        file_name = asksaveasfilename()
        file_name = file_name + ".avi"
        cap = cv2.VideoCapture(output_video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # fl

        file = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps / 3,
                               (width, height))
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                file.write(frame)
            else:
                break

        cap.release()
        file.release()
        cv2.destroyAllWindows()

    def extract_filed_from_dict(self, field):
        if self.state == play_original_video or self.state == general:
            return 0
        return self.gender_dict[field][self.current_frame]

    def play_processed_video(self):
        self.current_frame = 0
        self.is_should_init = True
        self.state = play_processed_video
        self.video_screen(output_video)

    def play_original_video(self):
        self.current_frame = 0
        self.is_should_init = True
        self.state = play_original_video
        self.video_screen(self.input_video)

    def video_screen(self, video_file):
        if self.is_should_init:
            self.canvas.delete("all")
            if video_file is not None:
                self.cap = cv2.VideoCapture(video_file)
        if video_file is not None:
            ret, frame = self.cap.read()
        males_count = self.extract_filed_from_dict("male")
        female_count = self.extract_filed_from_dict("female")
        leaving_targets = self.extract_filed_from_dict("outgoing_targets")
        incoming_targets = self.extract_filed_from_dict("current_targets")

        t = Label(self.root, text="Number of males: " + str(males_count),
                  font=("@Yu Gothic UI Semibold", 12, "bold"))
        t = Label(self.root, text="Number of males: " + str(males_count),
                  font=("@Yu Gothic UI Semibold", 12, "bold"))

        g = Label(self.root, text="Number of females: " + str(female_count),
                  font=("@Yu Gothic UI Semibold", 12, "bold"))

        f = Label(self.root, text="Number of targets in tracking: " + str(incoming_targets),
                  font=("@Yu Gothic UI Semibold", 12, "bold"))

        m = Label(self.root, text="Number of outgoing targets: " + str(leaving_targets),
                  font=("@Yu Gothic UI Semibold", 12, "bold"))

        y = Button(self.root, text="Play original video", fg="green",
                   font=("@Yu Gothic UI Semibold", 12, "bold"), command = self.play_original_video)

        r = Button(self.root, text="Play processed video",fg="green",
                   font=("@Yu Gothic UI Semibold", 12, "bold"), command = self.play_processed_video)

        e = Button(self.root, text="Save video", fg="orange",
                   font=("@Yu Gothic UI Semibold", 12, "bold"), command=self.save_as_file)

        s = Button(self.root, text="Start over", fg="red", font=("@Yu Gothic UI Semibold", 12, "bold"),
                   command=self.return_to_First_Screen)

        self.canvas.create_window(250, 50,
                                  window=t,
                                  anchor='nw')

        self.canvas.create_window(425, 50,
                                  window=g,
                                  anchor='nw')

        self.canvas.create_window(600, 50,
                                  window=f,
                                  anchor='nw')

        self.canvas.create_window(850, 50,
                                  window=m,
                                  anchor='nw')

        self.canvas.create_window(20, 200,
                                  window=y,
                                  anchor='nw')

        self.canvas.create_window(20, 300,
                                  window=r,
                                  anchor='nw')

        self.canvas.create_window(20, 400,
                                  window=e,
                                  anchor='nw')

        self.canvas.create_window(20, 500,
                                  window=s,
                                  anchor='nw')
        self.root.update()
        self.is_should_init = False

        if video_file is not None and ret:
            self.current_frame += 1
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            img = img.resize((1200, 600), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lmain = Label(self.root)
            self.lmain.imgtk = imgtk
            self.lmain.configure(image=imgtk)
            self.canvas.create_window(200, 150,
                                      window=self.lmain,
                                      anchor='nw')
            self.lmain.after(1, self.video_screen(video_file))


app = App()
