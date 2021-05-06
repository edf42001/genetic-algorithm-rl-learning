#!/bin/bash

# Uploads results of training runs to AWS S3 bucket
RUN_LOCATION="saved_runs"
RUN_NAME=$(ls ${RUN_LOCATION})

echo "Copying ${RUN_LOCATION}/${RUN_NAME} to S3 at s3://${S3_PATH}/${RUN_NAME} region ${AWS_REGION} ..."
aws s3 cp --recursive ${RUN_LOCATION}/${RUN_NAME} s3://${S3_PATH}/${RUN_NAME} --region ${AWS_REGION}
