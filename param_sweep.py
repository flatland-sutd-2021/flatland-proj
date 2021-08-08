import boto3

AWS_DEFAULT_REGION="ap-southeast-1"

GETTING_CLOSER 		= [0.0, -0.35, -0.5, -1.0, -5.0]
GETTING_FURTHER 	= [0.0, -0.35, -0.5, -1.0, -5.0]
FINAL_INCOMPLETE 	= [0, -50, -100, -500]

N_EPISODES = 800
JOB_QUEUE_NAME = "flatland-train-batch-job-queue"
JOB_DEFINITION_NAME = "flatland-train"

client = boto3.client('batch')

for closer_penalty in GETTING_CLOSER:
    for further_penalty in GETTING_FURTHER:
        for incomplete_penalty in FINAL_INCOMPLETE:
            job_name = f"eps-{N_EPISODES}-closer-{closer_penalty}-further-{further_penalty}-incomplete-{incomplete_penalty}"
            job_name = job_name.replace(".", "_")
            client.submit_job(
                jobName=job_name,
                jobQueue=JOB_QUEUE_NAME,
                jobDefinition=JOB_DEFINITION_NAME,
                containerOverrides={
                    "environment": [
                        {"name": "N_EPISODES", "value": str(N_EPISODES)},
                        {"name": "AWS_DEFAULT_REGION", "value": AWS_DEFAULT_REGION},
                        {"name": "REWARD_GETTING_CLOSER", "value": str(closer_penalty)},
                        {"name": "REWARD_GETTING_FURTHER", "value": str(further_penalty)},
                        {"name": "REWARD_FINAL_INCOMPLETE", "value": str(incomplete_penalty)}
                    ]
                }
            )
