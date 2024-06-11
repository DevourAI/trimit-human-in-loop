# trimit-human-in-loop

## Dev setup

1. Get copies of .dev/.env, .local/.env, and potentially .staging/.env, .prod/.env
1. Install mongodb-community@6.0 via homebrew
1. Stop the service if already started: `brew services stop mongodb/brew/mongodb-community@6.0`
1. Add the following lines to /opt/homebrew/etc/mongod.conf:
    ```
    replication:
      replSetName: "rs0"
    ```
1. Start the service again: `brew services start mongodb/brew/mongodb-community@6.0`
1. Run `mongosh` and then `rs.initiate()` to start a replica set
1. Use `mongodb://localhost:27017/?replicaSet=rs0` for MONGO_URL in .env.local
1. [Install poetry](https://python-poetry.org/docs/#installing-with-pipx)
1. `poetry install`
1. `poetry run pip install -e .`
1. `cd trimit/frontend && yarn install`


## Local frontend dev
```sh
local_ui_remote_backend
```

It's possible to run the ui using a local fastapi backend too:
```sh
make local_api
```

```sh
make local_ui
```

There are still some outstanding issues to make this local_api work as expected though

## Deploy server to dev

1. Do this the first time you deploy:
    ```sh
    poetry run modal token new
    ```
1. ```make deploy_dev```
