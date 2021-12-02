# Predicting Wine Quality using RF model in Spark trained over Cluster and dockerized

## Github
Parallel Training - https://github.com/navneet-kala/winequalitypredictor/blob/main/Train.py 
Single Machine Prediction -https://github.com/navneet-kala/winequalitypredictor/blob/main/Predict.py 

## Docker
https://hub.docker.com/repository/docker/navneetkala/winepredictnew

### To test docker on EC2
Go to ec2-instance and install docker and then
o	sudo service docker start
o	docker pull navneetkala/winepredict:latest
Please make sure that the test.csv file is available at /home/ec2-user, then run the below command replacing the filename below with your filename
sudo docker run -it -v `pwd`/TEST.csv:/home/ec2-user/TEST.csv navneetkala/winepredict /home/ec2-user/TEST.csv

### To test docker on Windows
o	Install docker desktop for Windows
o	Launch Powershell
o	docker pull navneetkala/winepredict:latest
Please make sure that the test.csv file is available at an example path like below ,then run the below command replacing the fileame below with your filename, also change the folder path accordingly
docker run -it -v C:\Users\19544\Downloads\TEST.csv:/TEST.csv navneetkala/winepredict /TEST.csv
