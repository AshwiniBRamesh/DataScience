from kafka import KafkaProducer
import json
import time
import random
from datetime import datetime,timedelta


def cust_movement():
	cust_id=random.randint(1, 200)
	cust_time=datetime.now()-timedelta(minutes=random.randint(1,20))
	cust_long=random.randrange(77,79)
	cust_lat=random.randrange(16,18)
	return cust_id,cust_time,cust_long,cust_lat

def json_serializer(data):
	return json.dumps(data,default=defaultconverter).encode("utf-8")

def defaultconverter(o):
  if isinstance(o, datetime):
      return o.__str__()

producer = KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=json_serializer)

if __name__ == "__main__":
	while 1==1 :
		cust_data = cust_movement()
		producer.send("cust-movement",cust_data)
		time.sleep(4)