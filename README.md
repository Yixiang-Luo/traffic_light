# traffic_light


### Dataset  
The dataset used is from [OpenDataLab S2TLD](https://opendatalab.org.cn/OpenDataLab/S2TLD/cli/main).  
The dataset folder only contains `.txt` files suitable for training YOLO models. However, you can download the images from the website mentioned above. Each label file corresponds to its respective image by name.

How to download the dataset：
https://opendatalab.org.cn/OpenDataLab/S2TLD/cli/main
- `pip install openxlab`  # Install openxlab
- `pip install -U openxlab`  # Upgrade openxlab
- `openxlab login`  # Log in, enter corresponding AK/SK (Access Key/Secret Key)
- `openxlab dataset info --dataset-repo OpenDataLab/S2TLD`  # View dataset information and file list
- `openxlab dataset get --dataset-repo OpenDataLab/S2TLD`  # Download dataset
- `openxlab dataset download --dataset-repo OpenDataLab/S2TLD --source-path /README.md --target-path /path/to/local/folder`  # Download dataset files


### Method  
We use YOLOv8 for training the model. After training, the `.pt` file (best weights) is saved in `runs/detect/train3/weights`.  

### Running the Application  
You can run the `app.py` script, which automatically loads the best weights file for inference.  

### Testing Files  
Testing files are located in the `presentation_files` directory. You can upload `.mp4`, `.png`, or `.jpg` files to the web application for testing.


