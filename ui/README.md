# ui

The Angular front end for Beegent — a pure consumer of the API in `../api`, which is itself a pure
consumer of the pipeline. It never re-implements ranking, confidence or verification.

```bash
npm ci
npx ng serve      # :4200, proxies /api to the backend on :8000
npx ng build      # production build; FastAPI serves it from dist/ui/browser
```

Running it, choosing a backend, and the boundary rules are documented once, in the repo root:
[README](../README.md#a-web-ui-if-you-prefer-one) ·
[Backends](../docs/backends.md#choosing-a-backend-for-the-web-ui) ·
[CONTRIBUTING](../CONTRIBUTING.md#the-web-ui).
