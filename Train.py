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

#Read training dataset
df_pyspark = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').csv("/home/ec2-user/TrainingDataset.csv")

count = df_pyspark.count()

print("\nStart Training:\n")
df_pyspark.printSchema()

#'quality' is label column.Others are feature columns. 
featureColumns = [col for col in df_pyspark.columns if col != '""""quality"""""']


#Use VectorAssembler, which is a feature transformer that merges multiple columns into a vector column
feature_assembler = VectorAssembler(inputCols=featureColumns, outputCol='features')
transformData = feature_assembler.transform(df_pyspark)


#Used Random Forest model
rf = RandomForestClassifier(featuresCol='features', labelCol='""""quality"""""',numTrees=100, maxBins=500, maxDepth=28, seed=40, minInstancesPerNode=2)
rfModel = rf.fit(transformData)


#Uses MulticlassClassificationEvaluator, which is an evaluator for Multiclass Classification, which expects input columns: prediction, label, weight 
eval = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="f1")
rfTrainingPredictions = rfModel.transform(transformData)


#F1 score
f1 = eval.evaluate(rfTrainingPredictions)

print("\nTraining Done\n")

print("Error in Training is = %g" % (1.0 - f1))

print("\nF1 score for the traning data is = %g\n" % f1)

#Save the model
rf.write().overwrite().save("newmodel.model")
