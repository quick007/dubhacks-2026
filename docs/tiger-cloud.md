# Tiger Cloud setup

This is the historical WSL/CLI setup. For the current optional experiment-results
adapter, use [Experiment results integration](../apps/api/EXPERIMENT_RESULTS.md).
It consumes `TIGER_DATABASE_URL`, and leaves the core SQLite store independent.
The machine-specific `DATABASE_URL` statements below describe the earlier setup.

Tiger CLI is installed as `tiger`. On this WSL machine, version 0.26.0 was installed through the official Debian repository. PostgreSQL's `psql` client is also available.

The local CLI is authenticated, with `dubhacks-2026` selected as its default service. This is a free development service using shared CPU/memory in `us-east-1`. Database passwords use `~/.pgpass` with mode `600`; production-tagged services are protected from writes by the CLI's `read_only=prod` setting. These are machine-local settings; teammates must authenticate separately.

## Sign in and select a database

```sh
tiger auth login --headless
tiger auth status
tiger service list
tiger config set service_id <service-id>
```

The login command prints a browser URL and a temporary code. Authorize it in your browser. If you have multiple Tiger Cloud projects, select the intended project. CLI configuration lives in `~/.config/tiger/config.yaml`; authentication is stored in the system keychain or a restricted credentials file.

Cloud account authentication and database passwords are separate. For an existing service whose password has not been saved locally, save its current password through the hidden prompt:

```sh
tiger db save-password
```

For WSL installations without a working system keychain, configure `pgpass` storage before saving the database password:

```sh
tiger config set password_storage pgpass
tiger db save-password
```

Use an existing development service. If the project has no service, choose the service tier and region before provisioning one.

## Project environment

The root `.env` is gitignored and contains the selected service's full `DATABASE_URL`, with mode `600`. Both a Tiger CLI SQL query and a `psql` query using the saved URL succeeded; the latter confirmed TLS 1.3.

To refresh the connection string, use the URI returned by `tiger db uri --with-password`. That command includes the database password, so redirect or capture its output rather than posting it in chat or committing it. Tiger URIs include `sslmode=require`.

The checked-in `.env.example` reserves `DATABASE_URL` without a credential. The current Go API uses SQLite through `go-sqlite3`; it does not read `DATABASE_URL` or load `.env`. Connecting the app to Tiger requires a separate PostgreSQL persistence change.

## Verify access

```sh
tiger db query --read-only -c 'SELECT current_database(), current_user, now();'
tiger db psql
```

The first command confirms an authenticated SQL connection. `tiger db ping` checks server availability but does not replace an authenticated query.

## Official documentation

- [CLI quickstart](https://www.tigerdata.com/docs/get-started/quickstart/tiger-cli)
- [CLI command and environment reference](https://www.tigerdata.com/docs/reference/tiger-cloud/tiger-cli)
- [Connection details and client credentials](https://www.tigerdata.com/docs/integrate/find-connection-details)
- [CLI source and releases](https://github.com/timescale/tiger-cli)
