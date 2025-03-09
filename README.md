# Project
docker cp elderly_care_knowledge_base.json mongodb:/tmp/elderly_care_knowledge_base.json
docker exec -it mongodb mongoimport --db MLOPs --collection knowledge_base --file /tmp/elderly_care_knowledge_base.json --jsonArray
docker compose up --build
