# trimit-human-in-loop

## Dev setup

1. Get copies of .dev/.env, .env.local, and potentially .staging/.env, .prod/.env
1. [Install poetry](https://python-poetry.org/docs/#installing-with-pipx)
1. `poetry install`
1. `poetry run pip install -e .`
1. `cd trimit/frontend && yarn install`


## Local frontend dev

To start the local api, run:
```sh
make local_api
```

And the local ui:
```
make local_ui
```

## Deploy server to dev

1. Do this the first time you deploy:
    ```sh
    poetry run modal token new
    ```
1. ```make deploy_dev```
