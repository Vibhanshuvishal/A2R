# Webhooks

Webhooks notify an endpoint when a subscribed event occurs. Configure an HTTPS endpoint, select event types, and verify signatures using the secret displayed at creation. Payloads include an event identifier and timestamp. The platform retries delivery after transient failures, so handlers must be idempotent and should acknowledge successful processing promptly.
