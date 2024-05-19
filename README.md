# trimit-human-in-loop

## Dev setup

1. Get copies of .dev/.env, .env.local, and potentially .staging/.env, .prod/.env
1. [Install poetry](https://python-poetry.org/docs/#installing-with-pipx)
1. `poetry install`
1. `poetry run pip install -e .`
1. `cd trimit/frontend && yarn install`


## Local frontend dev

To start the local server, run:
```sh
make local_webapp
```
