"""New occupants of a reused transport slot receive a full first snapshot."""
from std.testing import assert_equal, assert_true
from flint.app import AppConfig, EngineApp, STAGE_FIRST, STAGE_POST_UPDATE
from flint.core.math import Vec3
from flint.net.hub import NetHub
from flint.net.plugin import (
    NET_MODE_VIRTUAL, NetConfigResource, NetPlugin, NetReplicationResource, NetTransportResource,
    NetworkIdentityComponent, ReplicationComponent,
)
from flint.net.replicator import SnapshotClientReceiver
from flint.net.virtual_sim import VirtualEndpoint, VirtualNetSim
from flint.scene import GlobalTransform, LocalTransform, TransformPlugin


def rejoin_at_stage(stage: Int) raises:
    var sim = VirtualNetSim(seed=19, one_way_latency_ms=1, jitter_ms=0, loss_rate=0, dup_rate=0)
    var server = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0), True, max_channels=1)
    var peer = NetHub[VirtualEndpoint](sim.create_ipv4_peer(1), False)
    var endpoint = server.local_endpoint()
    # The old and returning client use the same address and transport slot.
    assert_equal(server.connect(peer.local_endpoint()), 0)
    server.drop(0)
    assert_equal(peer.connect(endpoint), 0)
    assert_true(peer.send_unreliable(0, List[UInt8](), sim.now_ms()))
    sim.advance_ms(5)

    var app = EngineApp(AppConfig(title="NetRejoin", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(NetPlugin(max_entities=8, max_clients=1))
    app.startup()
    ref cfg = app.resources.get_ptr_mut[NetConfigResource]().value()[]
    cfg.mode = NET_MODE_VIRTUAL
    cfg.snapshot_due = True
    app.resources.get_ptr_mut[NetTransportResource]().value()[].attach_virtual_hub(server^)
    var e = app.world.spawn()
    _ = app.world.insert[LocalTransform](e, LocalTransform(Vec3(7, 0, 0)))
    _ = app.world.insert[GlobalTransform](e, GlobalTransform())
    _ = app.world.insert[NetworkIdentityComponent](e, NetworkIdentityComponent(net_id=1))
    _ = app.world.insert[ReplicationComponent](e, ReplicationComponent())

    ref rep = app.resources.get_ptr_mut[NetReplicationResource]().value()[].replicator
    rep.set_entity(1, Vec3(7, 0, 0))
    rep.prepare_tick()
    var old = rep.build_client_packet(0)
    rep.notify_client_ack(0, old[0])
    rep.set_client_acked_input_seq(0, 987)
    assert_true(rep.clients[0].has_acked_seq)
    var prior_bytes = rep.clients[0].bytes_sent_total
    app.schedule.run_stage(stage, app.world, app.resources, app.pool, app.commands)
    assert_true(app.resources.get_ptr[NetTransportResource]().value()[].virtual_hubs[0].is_open(0))
    rep.notify_client_ack(0, old[0])  # A late ACK of the old peer's sequence must be ignored.
    assert_true(not rep.clients[0].has_acked_seq and not rep.clients[0].has_acked_input_seq)
    assert_equal(rep.clients[0].bytes_sent_total, prior_bytes)
    var first = rep.build_client_packet(0)
    assert_true(first[0] != old[0])
    var receiver = SnapshotClientReceiver(max_entities=8, world_bounds=512, chunk_size=64)
    assert_true(receiver.apply_packet(first[1])[0])
    assert_true(receiver.entities[1].active)
    assert_true(abs(receiver.entities[1].pos.x - Float32(7)) < 0.02)
    app.shutdown()


def main() raises:
    rejoin_at_stage(STAGE_FIRST)
    rejoin_at_stage(STAGE_POST_UPDATE)
    print("[PASS] Reused transport slot starts with a fresh baseline at both NetPlugin poll stages")
