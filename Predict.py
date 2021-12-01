from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
import sys
import pyspark.sql.functions as func
import pyspark

spark = SparkSession.builder.getOrCreate()
#spark.sparkContext.setLogLevel("ERROR")
#Single machine prediction application
conf = SparkConf().setAppName("Wine_Quality_Pred").setMaster("local[1]")

#sc = SparkContext.getOrCreate()
#sc.setLogLevel("ERROR")
#log4j = sc._jvm.org.apache.log4j
#log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)
import sys
fileName = sys.argv[1]

#Load trained model
#rf = RandomForestClassifier.load("wineQpredmodel.model")
rf = RandomForestClassifier.load("hdfs:////home/ec2-user/newmodel.model")


#Read the input data from csv
df_pyspark = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').load(fileName)
#df_pyspark = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').csv("hdfs:////home/ec2-user/ValidationDataset.csv")


#Assign all columns other than 'quality' as the feature columns. 'quality' is lable column.
featureColumns = [col for col in df_pyspark.columns if col != '""""quality"""""']


#Use VectorAssembler, which is a feature transformer that merges multiple columns into a vector column
feature_assembler = VectorAssembler(inputCols=featureColumns, outputCol='features')


#Use Pipeline, A Pipeline chains multiple Transformers and Estimators together to specify an ML workflow
spark_Pipe = Pipeline(stages=[feature_assembler, rf])

fitData = spark_Pipe.fit(df_pyspark)
transformedData = fitData.transform(df_pyspark)
transformedData = transformedData.withColumn("prediction", func.round(transformedData['prediction']))
transformedData = transformedData.withColumn('""""quality"""""', transformedData['""""quality"""""'].cast('double')).withColumnRenamed('""""quality"""""', "label")

results = transformedData.select(['prediction', 'label'])
predictionAndLabels = results.rdd

#Use MulticlassMetrics, evaluator for multiclass classification
rf_metrics = MulticlassMetrics(predictionAndLabels)

#Calculate precision, recall and F1 score (accuracy)
precision = rf_metrics.weightedPrecision
recall = rf_metrics.weightedRecall
f1Score = rf_metrics.weightedFMeasure()

print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
print("Final result/statistics: ")
print("=============================")
print("F1 Score  = %s" % f1Score)
print("=============================")
print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
