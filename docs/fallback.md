# Fallback models and credentials

`genai_agents::fallback::Fallback` is a `Model` of routes: each request goes to the first route that can take it.

```rust,ignore
use genai_agents::fallback::{Fallback, FallbackConfig};

let model = Fallback::new(OpenAiResponses::new(client_with_key_a, template.clone()))
    .credential(OpenAiResponses::new(client_with_key_b, template.clone())) // same model, another key
    .then(OpenAiChat::new(azure_client, azure_template));                  // another provider
let runner = Runner::new(model, tools);
```

Routes come in groups. `credential` adds the same model with other credentials to the last group; `then` starts a new
group: another model or provider, for when the groups before cannot answer.

| The route fails with | Then |
| --- | --- |
| refused or expired credentials (`ErrorKind::Authentication`) | it rests for as long as the provider asked, or `auth_rest` (10 minutes); the next credential takes the request |
| a rate limit or a server failure, with a wait the provider asked for | it rests that long; the next route takes the request |
| the same, without a wait | the runner retries; after `max_failures` (2) in a row it rests `failure_rest` (30 s), and the next route takes the request |
| an unknown model (`ErrorKind::ModelNotFound`) | its whole group is left for good |
| a local fault, a rejected request, a context too long | no fallback: the next route would fail the same way |

The route that answered last stays in use (`Fallback::current`): a provider keeps a prompt cache per account, which
another credential does not share. To keep a session on its route across processes, store `current()` with it and
give it back with `prefer`. When every route rests, a request fails with a retryable error, and the runner's retries
wait for one to come back.

Only failures before the response's body fall back. A stream that breaks half-way is the runner's to retry, on the
same route.

The runner reads the model's `info()` from the route in use, so the [projection](runner.md) leaves out the reasoning
another route's model wrote.
