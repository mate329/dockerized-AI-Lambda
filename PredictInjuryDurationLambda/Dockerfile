# Use an official AWS Lambda base image for Python
FROM public.ecr.aws/lambda/python:3.12

# Install required Python libraries
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the saved model and Lambda function code into the container
COPY lambda_handler.py ./
COPY injury_model.pkl ./
COPY model_metadata.json ./

# Command to run the Lambda function
CMD ["lambda_handler.lambda_handler"]
