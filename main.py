"""Getting data from Cassandra and sending to Kafka"""

import pandas as pd
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from pyspark.sql import SparkSession

CHANNEL_ID = "6215-9148"
TIME = "2018-8"
LIMIT = 15
CLUSTER_ID = ["10.150.73.157"]

ADDRESS = ["10.150.73.139:9092"]
TOPIC_NAME = "magda"

USERNAME = "eric"
PASSWORD = "cartman123"

DATA_PATH = "data/data.csv"


def get_data_from_cassandra(channel_id, time, limit, cluster_id):
    """ Get data from Cassandra """
    cluster = Cluster(
        cluster_id,
        auth_provider=PlainTextAuthProvider(
            username=USERNAME, password=PASSWORD),
    )
    session = cluster.connect()
    rows = session.execute(
        "SELECT * FROM production.data2 where channel='{}' and month='{}' limit {}".format(
            channel_id, time, limit
        )
    )
    data = pd.DataFrame(rows)
    return data


def send_data_kafka(topic_name, data, address):
    """ Send data to Kafka topic """
    admin_client = KafkaAdminClient(bootstrap_servers=address)
    producer = KafkaProducer(bootstrap_servers=address)
    consumer = KafkaConsumer(bootstrap_servers=address)
    if topic_name not in consumer.topics():
        new_topic = [
            NewTopic(name=topic_name, num_partitions=1, replication_factor=1)]
        admin_client.create_topics(new_topics=new_topic, validate_only=False)
    for row in data.collect():
        print(row)
        producer.send(topic_name, value=bytearray(str(row.asDict()), "utf-8"))
    producer.close()


def main():
    """Getting data from Cassandra and sending to Kafka"""

    # df = get_data_from_cassandra(CHANNEL_ID, TIME, LIMIT, CLUSTER_ID)
    # df.to_csv(DATA_PATH, index=False)

    spark = SparkSession.builder.master("local").appName("Kafka").getOrCreate()
    data = spark.read.csv(DATA_PATH, inferSchema=True, header=True)
    print("data read")
    send_data_kafka(TOPIC_NAME, data, ADDRESS)
    print("data sent")


if __name__ == "__main__":
    main()
