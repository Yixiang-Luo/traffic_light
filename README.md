# traffic_light


### Dataset  
The dataset used is from [OpenDataLab S2TLD](https://opendatalab.org.cn/OpenDataLab/S2TLD/cli/main).  

### Method  
We use YOLOv8 for training the model. After training, the `.pt` file (best weights) is saved in `runs/detect/train3/weights`.  

### Running the Application  
You can run the `app.py` script, which automatically loads the best weights file for inference.  

### Testing Files  
Testing files are located in the `presentation_files` directory. You can upload `.mp4`, `.png`, or `.jpg` files to the web application for testing.


