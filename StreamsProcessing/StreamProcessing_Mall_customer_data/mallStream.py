import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import math
import string
import random
import findspark
from pyspark import SparkConf, SparkContext
import pandas as pd
sc = SparkContext(master="local",appName="Spark Demo")


findspark.init()

KAFKA_INPUT_TOPIC_NAME_CONS = "cust-movement"
KAFKA_OUTPUT_TOPIC_NAME_CONS = "potential-cust"
KAFKA_BOOTSTRAP_SERVERS_CONS = 'localhost:9092'
MALL_LONGITUDE=78.446841
MALL_LATITUDE=17.427229
MALL_THRESHOLD_DISTANCE=100

if __name__ == "__main__":
	print("PySpark Structured Streaming with Kafka Application Started …")
	
	spark = SparkSession.builder.master("local").getOrCreate()
	sc.setLogLevel("ERROR")

	#cust_existing_data = spark.read.csv("C:\Users\ashwi\Downloads\pizzacustomers.csv")
	#cust_existing_data = spark.read.format("org.apache.spark.sql.csv").load(r'C:\Users\ashwi\Downloads\pizzacustomers.csv')
	#cust_existing_data = spark.read.option("header",True).csv("file:///C:\Users\ashwi\Downloads\pizzacustomers.csv').first()
	#PATH=u'C:\\Users\\ashwi\\Downloads\\pizzacustomers.csv'
	#cust_existing_data = spark.read.csv(PATH,header="true",inferSchema="true")
	#cust_existing_data.printSchema()
	
	#Import data from MySQL table into spark dataframe
	#customer_data_df=spark.sql("SELECT * FROM SPA_ASSGN_WAREHOUSE.MYMALL_CUSTOMER_DETAILS")
	
	pandasDF = pd.read_csv(r'C:\Users\ashwi\Downloads\pizza_customers.csv')
	pandasDF = pandasDF.rename(columns={"Annual Income (k$)":"Annual Income","Spending Score (1-100)":"Spending Score"})
	#Create PySpark DataFrame from Pandas
	sparkDF=spark.createDataFrame(pandasDF) 
	print("Printing Schema of sparkDF: ")
	sparkDF.printSchema()
	#sparkDF.show()


	stream_detail_df = spark \
	.readStream \
	.format("kafka") \
	.option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS_CONS) \
	.option("subscribe", KAFKA_INPUT_TOPIC_NAME_CONS) \
	.option("startingOffsets", "latest") \
	.load()

	#stream_detail_df = stream_detail_df.selectExpr("CAST(key AS STRING)","CAST(value AS STRING)")
	

	schema = StructType().add("a", IntegerType()).add("b", StringType())
	stream_detail_df.select( \
 	col("key").cast("string"),
  	from_json(col("value").cast("string"), schema))

	split_col = split(stream_detail_df['value'], ',')

	stream_detail_df = stream_detail_df.withColumn('Cust_id', split_col.getItem(0))

	stream_detail_df = stream_detail_df.withColumn('Vist_time', split_col.getItem(1))

	stream_detail_df = stream_detail_df.withColumn('Latitude', split_col.getItem(2).cast("float"))
	stream_detail_df = stream_detail_df.withColumn('Longitude', split_col.getItem(3).cast("float"))

	print("Printing Schema of stream_detail_df: ")
	stream_detail_df.printSchema()
	
	#Obtaining distance in meters using longitude and latitude
	stream_detail_df = stream_detail_df.withColumn('a', (
	pow(sin(radians(col("Latitude") - lit(MALL_LATITUDE)) / 2), 2) +
	cos(radians(lit(MALL_LATITUDE))) * cos(radians(col("Latitude"))) *
	pow(sin(radians(col("Longitude") - lit(MALL_LONGITUDE)) / 2), 2)
	)).withColumn("distance", atan2(sqrt(col("a")), sqrt(-col("a") + 1)) * 12742000)



	#Filtering customers based on distance
	stream_detail_df = stream_detail_df.drop("a")
	stream_detail_df = stream_detail_df.filter(col("distance") <= MALL_THRESHOLD_DISTANCE)

	#Joining Customer stream data with customer dataset
	stream_detail_df = stream_detail_df.join(sparkDF,stream_detail_df.Cust_id == sparkDF.CustomerID)
	print("Printing Schema of combined DataFrames: ")
	stream_detail_df.printSchema()


	#Customer Segmentation

	stream_detail_df = stream_detail_df.withColumn("Customer_Segment", when( (col("Spending Score") > 60) & (col("Annual Income") <40000), 'Sleepers').when( ((col("Spending Score") >= 40) & (col("Spending Score") <= 60)) & ((col("Annual Income") >= 40000)&(col("Annual Income")<=70000)), 'Regulars').when( (col("Spending Score") > 60) & (col("Annual Income") >70000), 'Champions').when( (col("Spending Score") > 60) & (col("Annual Income") <40000), 'Loyals').when( (col("Spending Score") < 40) & (col("Annual Income") <40000), 'At Risk'))

	# Output topic dataframe creation by selecting required columns

	final_stream_df = stream_detail_df.selectExpr("CustomerID","Customer_Segment")

	final_stream_df = final_stream_df.withColumn("key",rand()*3)
	# Write key-value data from a DataFrame to a specific Kafka topic specified in an option
	customer_detail_write_stream_1 = final_stream_df \
	.selectExpr("CAST(key AS STRING)", "to_json(struct(*)) AS value") \
	.writeStream \
	.format("kafka") \
	.option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS_CONS) \
	.option("topic", KAFKA_OUTPUT_TOPIC_NAME_CONS) \
	.trigger(processingTime='1 seconds') \
	.outputMode("update") \
	.option("checkpointLocation", "C:\Python376\Ashwini_codes") \
	.start()

	#customer_detail_write_stream.awaitTermination()

	print("PySpark Structured Streaming with Kafka Application Completed.")



