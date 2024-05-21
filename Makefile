.PHONY: deploy_prod \
	deploy_dev \
	deploy_staging \
	test \
	upload_test_fixtures \
	format \
	local_webapp \
	setup-hooks \
	integration_test \
	integration_test_with_deploy \
	step_ephemeral

deploy_prod:
	@DEPLOY_BACKEND=true DEPLOY_FRONTEND=true ENV=prod ./deploy.sh

deploy_staging:
	@DEPLOY_BACKEND=true DEPLOY_FRONTEND=true ENV=staging ./deploy.sh

deploy_dev:
	@DEPLOY_BACKEND=true DEPLOY_FRONTEND=true ENV=dev ./deploy.sh

test:
	@echo "Running tests..."
	ENV=test poetry run pytest -v tests $(TEST_ARGS)

integration_test:
	@echo "Running integration tests..."
	ENV=dev poetry run pytest -v integration_tests $(TEST_ARGS)

integration_test_with_deploy:
	@echo "Running integration tests..."
	@ENV=dev make deploy_dev && poetry run pytest -v integration_tests $(TEST_ARGS)

setup-hooks:
	@echo "Setting up git hooks..."
	@./scripts/setup-hooks.sh

format:
	@poetry run black .

local_api:
	@echo "Starting local api..."
	@ENV=dev VERCEL_FRONTEND_URL="http://127.0.0.1:3000" poetry run python -m uvicorn trimit.api.index:web_app --reload

local_ui:
	@echo "Starting local ui..."
	@cd trimit/frontend && yarn dev

step_ephemeral:
	@echo "Running step function locally..."
	@poetry run modal run trimit.backend.serve $(STEP_ARGS)
