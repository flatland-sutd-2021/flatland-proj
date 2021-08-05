IMAGE_NAME="flatland-train"

REGISTRY_ID=$(aws sts get-caller-identity --query 'Account' --output text)
REGION=$(aws configure get region)

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com
docker build -t ${IMAGE_NAME} .
docker tag ${IMAGE_NAME}:latest ${REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:latest
docker push ${REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:latest
