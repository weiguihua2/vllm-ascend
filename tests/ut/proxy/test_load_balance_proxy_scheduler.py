import asyncio
import hashlib

import pytest

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy


def test_prefill_session_affinity_survives_load_changes():
    scheduler = proxy.SharedProxyScheduler(
        prefiller_instances=[("127.0.0.1", 19001), ("127.0.0.1", 19002)],
        decoder_instances=[("127.0.0.1", 19003)],
    )
    first = scheduler.begin_request(1.0, "trajectory-1")
    # A fresh request changes the load ordering, but later turns in this
    # trajectory must still reach the P instance holding its cached prefix.
    scheduler.begin_request(10_000.0)
    later = scheduler.reserve_prefill_kv(1.0, "trajectory-1")
    assert later["key"] == first["key"]


def test_prefill_request_pins_dp_rank():
    class Response:
        def raise_for_status(self):
            pass

    class Client:
        headers = None

        async def post(self, endpoint, *, json, headers):
            self.headers = headers
            return Response()

    client = Client()
    asyncio.run(
        proxy.send_request_to_service(
            client,
            "/v1/chat/completions",
            {"messages": []},
            "req-1",
            affinity_key="trajectory-1",
            prefill_dp_size=4,
        )
    )
    expected = int.from_bytes(hashlib.sha256(b"trajectory-1").digest()[:8], "big") % 4
    assert client.headers["X-Correlation-ID"] == "trajectory-1"
    assert client.headers["X-data-parallel-rank"] == str(expected)


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("retry prefill failed"), asyncio.CancelledError()],
    ids=["prefill-error", "request-cancelled"],
)
def test_failed_reassignment_releases_previous_decoder_once(monkeypatch, failure):
    scheduler = proxy.SharedProxyScheduler(
        prefiller_instances=[("127.0.0.1", 19001)],
        decoder_instances=[("127.0.0.1", 19002)],
    )
    runtime = proxy.WorkerRuntime(scheduler)
    monkeypatch.setattr(proxy, "runtime", runtime)

    prefiller_score = 100.0
    decoder_score = 1000.0
    prefiller = scheduler.begin_request(prefiller_score)
    decoder = scheduler.pick_decoder(decoder_score)
    previous_instance = proxy.InstanceInfo(
        request_id="request-0",
        prefiller_key=prefiller["key"],
        prefiller_score=prefiller_score,
        decoder_key=decoder["key"],
        decoder_score=decoder_score,
        decoder_host=decoder["host"],
        decoder_port=decoder["port"],
    )

    async def fail_assignment(*args, **kwargs):
        raise failure

    monkeypatch.setattr(proxy, "assign_instances", fail_assignment)

    async def run_reassignment_and_cleanup():
        with pytest.raises(type(failure)):
            await proxy.reassign_instances(
                "/v1/chat/completions",
                {},
                128,
                previous_instance,
                previous_prefiller_kv_released=True,
            )
        await proxy._finish_instance(runtime, previous_instance, release_prefill_kv=False)

    asyncio.run(run_reassignment_and_cleanup())

    decoder_state = scheduler.decoders[decoder["key"]]
    assert decoder_state.active_tokens == 0.0
    assert scheduler.request_num == 0
