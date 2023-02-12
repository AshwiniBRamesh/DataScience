from kafka import KafkaConsumer
from json import loads
import time

consumer = KafkaConsumer("potential-cust",bootstrap_servers=['localhost:9092'],enable_auto_commit=True)

if __name__=="__main__":
	message_list=[]
	for message in consumer:
		print(message.key, message.value)
		message_list.append(message.value)
		time.sleep(4)
