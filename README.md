# FedCCRL: Federated Domain Generalization with Cross-Client Representation Learning
# Dataset Preparation
download dataset and put them at data/[dataset]. The structure should be data/[dataset_name]/raw/domain_name/label/image.
# How to Run
You can run main.py directly with 
~~~
python main.py [Algorithm] -d [Dataset Name] [other arguments] 
~~~
Example：
~~~
python main.py FedCCRL -d minidomainnet 
~~~
