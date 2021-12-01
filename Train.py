from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics

import pyspark.sql.functions as func
import pyspark

spark = SparkSession.builder.getOrCreate()

#Run locally with 4 cores. Parallel training implementation
#conf = SparkConf().setAppName("Wine_Quality_Pred").setMaster("local[4]")

sc = SparkContext.getOrCreate()
#sc._jsc.hadoopConfiguration().set("fs.s3a.awsAccessKeyId", 'AASIAUSW3OPQ4CH4QP6PU')
#sc._jsc.hadoopConfiguration().set("fs.s3a.awsSecretAccessKey", 'hMGra5+Genp95VMeNREw/JXcWTeEP+WTEc2YrDlU')
#Read training dataset from S3 bucket
df_pyspark = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').csv("hdfs:////home/ec2-user/TrainingDataset.csv")

count = df_pyspark.count()

print("\nTraining Schema:\n")
df_pyspark.printSchema()

#Assign all columns other than 'quality' as the feature columns. 'quality' is lable column.
featureColumns = [col for col in df_pyspark.columns if col != '""""quality"""""']


#Use VectorAssembler, which is a feature transformer that merges multiple columns into a vector column
feature_assembler = VectorAssembler(inputCols=featureColumns, outputCol='features')
transformData = feature_assembler.transform(df_pyspark)


#Train Random Forest model
rf = RandomForestClassifier(featuresCol='features', labelCol='""""quality"""""',numTrees=100, maxBins=500, maxDepth=28, seed=40, minInstancesPerNode=2)
rfModel = rf.fit(transformData)


#Use MulticlassClassificationEvaluator, which is an evaluator for Multiclass Classification, which expects input columns: prediction, label, weight 
eval = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="f1")
rfTrainingPredictions = rfModel.transform(transformData)


#Calculate F1 score or accuracy
f1 = eval.evaluate(rfTrainingPredictions)

print("\n *** Model Training Completed ***\n")

print("Training error = %g" % (1.0 - f1))

print("\nRandom Forest F1 score of traning data = %g\n" % f1)

#Save the trained model
#rf.write().save("s3a://mlmodel.cs643/model/newmodel.model")
rf.write().overwrite().save("hdfs:////home/ec2-user/newmodel.model")
