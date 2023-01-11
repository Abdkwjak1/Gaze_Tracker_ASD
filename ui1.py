# -*- coding: utf-8 -*- 
""" 
Created on Wed Aug 25 13:00:47 2021 
 
@author: Rama 
""" 
 
import tkinter as tk 
import tkinter.font as tkFont 
import deepvog

 
class App: 
    def __init__(self, root):
        self.root=root 
        self.inferer = deepvog.gaze_inferer( 3.6, [240,320], (2.02, 3.58))
        #setting title 
        root.title("AR'S Eye Tracker") 
        #setting window size 
        width=603 
        height=393 
        screenwidth = root.winfo_screenwidth() 
        screenheight = root.winfo_screenheight() 
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2) 
        root.geometry(alignstr) 
        root.resizable(width=False, height=False) 
 
        L1 =tk.Label(root) 
        ft = tkFont.Font(family='Times',size=38) 
        L1["font"] = ft 
        L1["fg"] = "#7b6674" 
        L1["justify"] = "center" 
        L1["text"] = "AR'S Eye Tracker" 
        L1.place(x=90,y=10,width=385,height=55) 
 
        Bu1=tk.Button(root) 
        Bu1["bg"] = "#b7ddb7" 
        ft = tkFont.Font(family='Times',size=18) 
        Bu1["font"] = ft 
        Bu1["fg"] = "#571718" 
        Bu1["justify"] = "center" 
        Bu1["text"] = "Child Info" 
        Bu1.place(x=30,y=100,width=293,height=47) 
        Bu1["command"] = self.Bu1_command 
 
        Bu2=tk.Button(root) 
        Bu2["bg"] = "#85a78f" 
        ft = tkFont.Font(family='Times',size=18) 
        Bu2["font"] = ft 
        Bu2["fg"] = "#571718" 
        Bu2["justify"] = "center" 
        Bu2["text"] = "Run Test" 
        Bu2.place(x=280,y=170,width=293,height=47) 
        Bu2["command"] = self.Bu2_command 
 
        Bu3=tk.Button(root) 
        Bu3["bg"] = "#e8d870" 
        ft = tkFont.Font(family='Times',size=18) 
        Bu3["font"] = ft 
        Bu3["fg"] = "#571718" 
        Bu3["justify"] = "center" 
        Bu3["text"] = "Check System" 
        Bu3.place(x=30,y=240,width=293,height=47) 
        Bu3["command"] = self.Bu3_command 
 
        Bu4=tk.Button(root) 
        Bu4["bg"] = "#c2d7d5" 
        ft = tkFont.Font(family='Times',size=18) 
        Bu4["font"] = ft 
        Bu4["fg"] = "#571718" 
        Bu4["justify"] = "center" 
        Bu4["text"] = "Get Result" 
        Bu4.place(x=280,y=310,width=293,height=47) 
        Bu4["command"] = self.Bu4_command 
 
    def Bu1_command(self): 
        self.child_inf(root) 
 
 
    def Bu2_command(self): 
        self.run_test(root) 
 
 
    def Bu3_command(self): 
        self.inferer.process( video_src="live",mode="default",vis=True) 
 
 
    def Bu4_command(self): 
        print("command") 


    def child_inf(self,root1):
        root=tk.Toplevel(root1) 
        #setting title 
        root.title("CHILD INFO") 
        #setting window size 
        width=633 
        height=326 
        screenwidth = root.winfo_screenwidth() 
        screenheight = root.winfo_screenheight() 
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2) 
        root.geometry(alignstr) 
        root.resizable(width=False, height=False) 
 
        
        label1=tk.Label(root) 
        ft = tkFont.Font(family='Times',size=13) 
        label1["font"] = ft 
        label1["fg"] = "#571718" 
        label1["justify"] = "center" 
        label1["text"] = "Name :" 
        label1.place(x=10,y=20,width=101,height=45) 
 
        En1=tk.Entry(root) 
        En1["borderwidth"] = "1px" 
        ft = tkFont.Font(family='Times',size=10) 
        En1["font"] = ft 
        En1["fg"] = "#333333" 
        En1["justify"] = "center" 
        En1["text"] = "" 
        En1.place(x=130,y=20,width=481,height=41) 
 
        label2=tk.Label(root) 
        ft = tkFont.Font(family='Times',size=13) 
        label2["font"] = ft 
        label2["fg"] = "#571718" 
        label2["justify"] = "center" 
        label2["text"] = "Age :" 
        label2.place(x=10,y=80,width=101,height=45) 
 
        En2=tk.Entry(root) 
        En2["borderwidth"] = "1px" 
        ft = tkFont.Font(family='Times',size=10) 
        En2["font"] = ft 
        En2["fg"] = "#333333" 
        En2["justify"] = "center" 
        En2["text"] = "" 
        En2.place(x=130,y=80,width=481,height=41) 
 
        label3=tk.Label(root) 
        ft = tkFont.Font(family='Times',size=13) 
        label3["font"] = ft 
        label3["fg"] = "#571718" 
        label3["justify"] = "center" 
        label3["text"] = "Child Rank :" 
        label3.place(x=10,y=140,width=122,height=45) 
 
        En3=tk.Entry(root) 
        En3["borderwidth"] = "1px" 
        ft = tkFont.Font(family='Times',size=10) 
        En3["font"] = ft 
        En3["fg"] = "#333333" 
        En3["justify"] = "center" 
        En3["text"] = "" 
        En3.place(x=130,y=140,width=481,height=41) 
 
        label4=tk.Label(root) 
        ft = tkFont.Font(family='Times',size=13) 
        label4["font"] = ft 
        label4["fg"] = "#571718" 
        label4["justify"] = "center" 
        label4["text"] = "Case Level :" 
        label4.place(x=10,y=200,width=122,height=45) 
 
        En4=tk.Entry(root) 
        En4["borderwidth"] = "1px" 
        ft = tkFont.Font(family='Times',size=10) 
        En4["font"] = ft 
        En4["fg"] = "#333333" 
        En4["justify"] = "center" 
        En4["text"] = "" 
        En4.place(x=130,y=200,width=481,height=41) 
 
        label5=tk.Label(root) 
        ft = tkFont.Font(family='Times',size=13) 
        label5["font"] = ft 
        label5["fg"] = "#571718" 
        label5["justify"] = "center" 
        label5["text"] = "Phys Condition:" 
        label5.place(x=10,y=260,width=122,height=45) 
 
        En5=tk.Entry(root) 
        En5["borderwidth"] = "1px" 
        ft = tkFont.Font(family='Times',size=10) 
        En5["font"] = ft 
        En5["fg"] = "#333333" 
        En5["justify"] = "center" 
        En5["text"] = "" 
        En5.place(x=130,y=260,width=481,height=41)


    def run_test(self, root1):
        root=tk.Toplevel(root1) 
        #setting title 
        root.title("RUN TEST") 
        #setting window size 
        width=324 
        height=246 
        screenwidth = root.winfo_screenwidth() 
        screenheight = root.winfo_screenheight() 
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2) 
        root.geometry(alignstr) 
        root.resizable(width=False, height=False) 
 
        
        button1=tk.Button(root) 
        button1["bg"] = "#858585" 
        ft = tkFont.Font(family='Times',size=23) 
        button1["font"] = ft 
        button1["fg"] = "#a43e1f" 
        button1["justify"] = "center" 
        button1["text"] = "Calibration" 
        button1.place(x=20,y=30,width=282,height=40) 
        button1["command"] = self.button1_command 
 
        button2=tk.Button(root) 
        button2["bg"] = "#abcfab" 
        ft = tkFont.Font(family='Times',size=23) 
        button2["font"] = ft 
        button2["fg"] = "#a43e1f" 
        button2["justify"] = "center" 
        button2["text"] = "Infer Geze" 
        button2.place(x=20,y=100,width=282,height=40) 
        button2["command"] = self.button2_command 
 
        button3=tk.Button(root) 
        button3["bg"] = "#cf9078" 
        ft = tkFont.Font(family='Times',size=23) 
        button3["font"] = ft 
        button3["fg"] = "#a43e1f" 
        button3["justify"] = "center" 
        button3["text"] = "Procced" 
        button3.place(x=20,y=170,width=282,height=40) 
        button3["command"] = self.button3_command
 


    def button1_command(self): 
        self.inferer.process(video_src="live", mode="Fit",calibration_samples=300,vis=True) 
 
 
    def button2_command(self): 
        self.inferer.process(video_src="live", mode="Infer",vis=True) 
 
 
    def button3_command(self): 
        print("command")


if __name__ == "__main__": 
    root = tk.Tk() 
    app = App(root) 
    root.mainloop()