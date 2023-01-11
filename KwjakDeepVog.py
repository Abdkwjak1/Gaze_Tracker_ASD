# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:02:41 2021

@author: Abd_Kwjak
"""

import deepvog

def syst():
    print("..EyeTracking System is getting started..")
    print("""..Command List:
    ..Session mode: 'live' or video name with ext
    ..EyeTracking mode: 'default' or 'Fit' or 'Infer'
    ..Type 'Quit' to End Program
    ..Press Q To Quit single Session
             
    NOTE: You must do fit eye_ball model before infer gaze
             
             """)
    while True:    
        mode=""
        sess=""
        
        sess=input("Enter Session mode.. OR 'Quit':")
        if sess == "Quit":
            return
        else:
            mode= input("Enter EyrTracking mode:")
        
            # Initialize the class. It requires information of your camera's focal length and sensor size, which should be available in product manual.
            inferer = deepvog.gaze_inferer( 3.6, [240,320], (2.02, 3.58)) 
        
            if mode == "default":
                inferer.process( video_src=sess,mode=mode,vis=True)
            elif mode == "Fit":    
                # Fit an eyeball model from "demo.mp4". The model will be stored as the "inferer" instance's attribute.
                inferer.process(video_src=sess, mode=mode,calibration_samples=300,vis=True)

                #After fitting, infer gaze from "demo.mp4" and output the results into "demo_result.csv"
                inferer.process(video_src=sess, mode="Infer",vis=True)

syst()

# Optional

# You may also save the eyeball model to "demo_model.json" for subsequent gaze inference
#inferer.save_eyeball_model("demo_model.json") 

# By loading the eyeball model, you don't need to fit the model again
#inferer.load_eyeball_model("demo_model.json") 