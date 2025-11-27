default:
    @just --list
    
dev:
    docker compose -f compose.dev.yml up -d 
    
prod:
    docker compose -f compose.yml up -d 
    
stop:
    docker compose down
    
migrate:
    docker compose exec backend alembic upgrade head
    
generate-schema:
    cd ./backend && uv run strawberry export-schema app.graphql.schema --output backend/backend/schema.graphql
    
generate-types:
    cd ./web && npm run generate-types
    
backend-shell:
    docker compose exec -it backend bash
    
frontend-shell:
    docker compose exec -it frontend bash
    