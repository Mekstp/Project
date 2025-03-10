# Project
docker cp elderly_care_knowledge_base_full.json mongodb:/tmp/elderly_care_knowledge_base_full.json

docker exec -it mongodb mongoimport --db MLOPs --collection knowledge_base --file /tmp/elderly_care_knowledge_base_full.json --jsonArray

docker compose up --build
