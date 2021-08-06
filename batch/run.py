import os
import boto3

client = boto3.client('s3')

client.put_object(
    Key='test_object',
    Body=b'Hello!',
    Bucket='flatland-train-output',
)
