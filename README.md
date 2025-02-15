# crack-detector
A nice crack detector :)

How to run:

#### WebApp
- `python predict.py --meta_file="./alg1_output/model_complete.meta" --CP_dir="./alg1_output" --start_as_server=True`
- `python predict.py --meta_file="./alg2_output/model_complete.meta" --CP_dir="./alg2_output" --start_as_server=True --port=8081 --model_number=2`
- open WebApp/index.html in a browser disabling CORS, here is how to do it for chrome: https://alfilatov.com/posts/run-chrome-without-cors/

#### Project 1:
Based on https://www.researchgate.net/publication/315613676_Deep_Learning-Based_Crack_Damage_Detection_Using_Convolutional_Neural_Networks?fbclid=IwAR1NqL8c7qwKNCFxAW7E9BAW6c98DKCSEfdgoYAB0WYY5iaQVSTqxBMVqCY

Requirements:
- tensorflow: 1.15.0
- matplotlib: 3.1.2
- opencv-python: 4.1.2.30

How to run:

- Change directory to CracksDetectionApp (`cd CracksDetectionApp`)
- `python trainAlg1.py`
- `python trainAlg2.py`

How to test:

- Change directory to CracksDetectionApp (cd CracksDetectionApp)
- `python predict.py --meta_file="./alg1_output/model_complete.meta" --CP_dir="./alg1_output"`
- look into folder CracksDetectionApp/results for the resulting images 

In order to test the other algorithm run:
- `python predict.py --meta_file="./alg2_output/model_complete.meta" --CP_dir="./alg2_output"`

To start the predictor as servers for each algorithm run the following commands:
- `python predict.py --meta_file="./alg1_output/model_complete.meta" --CP_dir="./alg1_output" --start_as_server=True`
- `python predict.py --meta_file="./alg2_output/model_complete.meta" --CP_dir="./alg2_output" --start_as_server=True --port=8081 --model_number=2`


#### Project 2 Unet:
Based on https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47

Requirements:
- numpy	1.17.4
- pandas	0.25.3
- matplotlib	3.1.2
- Pillow	6.2.1
- tqdm	4.40.0
- scikit-image	0.16.2
- tensorflow	2.0.0

How to run:
- cd CommandLineApp/CrackDetection/
- python ImageProcessorUsingUNET.py

#### Project 3 V2:

Requirements:
- tensorflow	2.0.0
- matplotlib	3.1.2
- aspectlib	1.4.2

How to run:
- python CommandLineApp/CrackDetection/ImageProcessorV2.py

How to test:
- python CommandLineApp/CrackDetection/predictionV2.py

technical report link : https://docs.google.com/document/d/1XFI4-NG3JoQ1_NO4GCPsOfUy0MokdtFchDorpWL-vLI/edit?fbclid=IwAR0nZoZapOnE3ei6PkDeUQHi26dwDNDQne-qzYXeWncD7e6qYNYw2_YZ4h8

google docs link : https://docs.google.com/document/d/1HZ3XukJdGuqtobCRUvc7KHZCh3Oun3Pu6DqvPCUoquE/edit?usp=sharing

presentation link : https://docs.google.com/presentation/d/1DjbLFpMnE6zBXJQ2NxbrZQuKOJR_Lwlu5tKPKM8E8As/edit?fbclid=IwAR3YqGYsOe-zBtoFLWWVQR8RJFR6HVrujhF05Yt-fmvpuGBRf9jjZQ8cPVc#slide=id.p2

trello link : https://trello.com/b/60KTWa0j/crack-detector

Coordinators: Associate Professor, PhD Adrian Iftene
              PhD Anca Ignat
              
Members: 
- Razvan Nica (contact: razvan.nica30@gmail.com)
- Serban Mihai Botez (contact: serbanbotez2@gmail.com)
- Stefan Cosmin Romanescu (contact: stefancosmin.romanescu@gmail.com)
- Florin Cristian Finaru (contact: florincristian.finaru@gmail.com)
