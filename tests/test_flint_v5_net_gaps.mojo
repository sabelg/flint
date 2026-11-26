"""Flint v5 networking gap-closure suite (`flint.net` spawn / interpolation / prediction /
component replication + RPC / X25519+HKDF session security) over the deterministic `VirtualNetSim`."""
from std.collections import List
from std.math import abs
from std.testing import assert_equal, assert_true
from flint.app import AppConfig, EngineApp
from flint.core.math import Quat, Vec3
from flint.ecs.world import Entity, STORAGE_TABLE, World, comp_hash64
from flint.net import (
    AeadKey,
    BitReader,
    BitWriter,
    IPV6_SAFE_MTU_BYTES,
    NetHub,
    RollbackInput,
    VirtualEndpoint,
    VirtualNetSim,
)
from flint.net.plugin import (
    NET_AUTHORITY_CLIENT,
    NET_AUTHORITY_SERVER,
    NET_MODE_VIRTUAL,
    NET_PEER_ALL,
    NET_RPC_TARGET_NONE,
    NetConfigResource,
    NetPlugin,
    NetPodComponent,
    NetReplicationResource,
    NetTransportResource,
    NetworkIdentityComponent,
    PredictionResource,
    ReplicationComponent,
    pop_client_input_as,
    register_net_pod_component,
)
from flint.net.prediction import PredictedState
from flint.scene import GlobalTransform, LocalTransform, TransformPlugin
from flint.net.kex import (
    CONNECT_TOKEN_BYTES,
    derive_session_keys,
    hkdf_expand,
    hkdf_extract,
    hkdf_sha256,
    hmac_sha256,
    issue_connect_token,
    sha256,
    verify_connect_token,
    x25519,
    x25519_public_key,
)


def _hex(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    var b = s.as_bytes()
    var i = 0
    while i + 1 < len(b):
        var hi = Int(b[i])
        var lo = Int(b[i + 1])
        hi = hi - 48 if hi <= 57 else (hi - 87 if hi >= 97 else hi - 55)
        lo = lo - 48 if lo <= 57 else (lo - 87 if lo >= 97 else lo - 55)
        out.append(UInt8(hi * 16 + lo))
        i += 2
    return out^


def _eq(a: List[UInt8], b: List[UInt8]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _bytes_of(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    var b = s.as_bytes()
    for i in range(len(b)):
        out.append(b[i])
    return out^


def test_kex_vectors() raises:
    assert_true(_eq(sha256(_bytes_of("abc")), _hex("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")))
    assert_true(_eq(
        hmac_sha256(_bytes_of("key"), _bytes_of("The quick brown fox jumps over the lazy dog")),
        _hex("f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"),
    ))
    # RFC 5869 test case 1
    var ikm = List[UInt8](length=22, fill=UInt8(0x0B))
    var prk = hkdf_extract(_hex("000102030405060708090a0b0c"), ikm)
    assert_true(_eq(prk, _hex("077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5")))
    var okm = hkdf_expand(prk, _hex("f0f1f2f3f4f5f6f7f8f9"), 42)
    assert_true(_eq(okm, _hex("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865")))
    # RFC 5869 test case 3 (empty salt / info)
    var okm3 = hkdf_sha256(List[UInt8](), ikm, List[UInt8](), 42)
    assert_true(_eq(okm3, _hex("8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8")))
    print("[PASS] SHA-256 / HMAC-SHA256 / RFC 5869 HKDF-SHA256 test vectors")

    # RFC 7748 section 5.2
    assert_true(_eq(
        x25519(
            _hex("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4"),
            _hex("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c"),
        ),
        _hex("c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552"),
    ))
    assert_true(_eq(
        x25519(
            _hex("4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d"),
            _hex("e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493"),
        ),
        _hex("95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957"),
    ))
    # RFC 7748 section 6.1 Diffie-Hellman
    var a_priv = _hex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a")
    var b_priv = _hex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb")
    var a_pub = x25519_public_key(a_priv)
    var b_pub = x25519_public_key(b_priv)
    assert_true(_eq(a_pub, _hex("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a")))
    assert_true(_eq(b_pub, _hex("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f")))
    var k1 = x25519(a_priv, b_pub)
    var k2 = x25519(b_priv, a_pub)
    assert_true(_eq(k1, _hex("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742")))
    assert_true(_eq(k1, k2))
    var ks_a = derive_session_keys(k1, a_pub, b_pub)
    var ks_b = derive_session_keys(k2, a_pub, b_pub)
    assert_true(ks_a[0].w0 == ks_b[0].w0 and ks_a[1].w7 == ks_b[1].w7)
    assert_true(ks_a[0].w0 != ks_a[1].w0)
    print("[PASS] RFC 7748 X25519 vectors + HKDF per-direction session keys")

    var tk = _bytes_of("backend-secret-key")
    var tok = issue_connect_token(tk, 0x464C, 42, 5000, 7)
    assert_equal(len(tok), CONNECT_TOKEN_BYTES)
    var info = verify_connect_token(tk, tok, 0x464C, 100.0)
    assert_true(info.valid and info.client_id == 42 and info.user_data == 7)
    assert_true(not verify_connect_token(tk, tok, 0x464C, 6000.0).valid)  # expired
    assert_true(not verify_connect_token(tk, tok, 0x9999, 100.0).valid)  # wrong protocol
    var forged = tok.copy()
    forged[8] ^= 1
    assert_true(not verify_connect_token(tk, forged, 0x464C, 100.0).valid)
    assert_true(not verify_connect_token(_bytes_of("other"), tok, 0x464C, 100.0).valid)
    print("[PASS] HMAC-SHA256 connect token issue/verify/forgery/expiry")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_app(title: String) -> EngineApp:
    var app = EngineApp(AppConfig(title=title, headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(NetPlugin(max_entities=64, max_clients=4))
    app.startup()
    app.resources.get_ptr_mut[NetConfigResource]().value()[unsafe_offset=0].mode = NET_MODE_VIRTUAL
    return app^


def _attach(mut app: EngineApp, var hub: NetHub[VirtualEndpoint]):
    app.resources.get_ptr_mut[NetTransportResource]().value()[unsafe_offset=0].attach_virtual_hub(hub^)


def _find_net_entity(mut app: EngineApp, net_id: Int) -> Tuple[Bool, Entity]:
    var ents = app.world.query_entities(app.world.bit_for[NetworkIdentityComponent]())
    for i in range(len(ents)):
        var nc = app.world.get_ptr[NetworkIdentityComponent](ents[i])
        if nc and nc.value()[unsafe_offset=0].net_id == net_id:
            return (True, ents[i])
    return (False, Entity())


def _spawn_server_entity(
    mut app: EngineApp, net_id: Int, pos: Vec3, owner: Int = 0,
    authority: UInt8 = NET_AUTHORITY_SERVER, prefab: UInt32 = 0,
) -> Entity:
    var e = app.world.spawn()
    _ = app.world.insert[LocalTransform](e, LocalTransform(pos))
    _ = app.world.insert[GlobalTransform](e, GlobalTransform())
    _ = app.world.insert[NetworkIdentityComponent](
        e, NetworkIdentityComponent(net_id, owner, False, authority, prefab)
    )
    _ = app.world.insert[ReplicationComponent](e, ReplicationComponent())
    return e


def _spawn_prefab7(world_ptr: Pointer[World, MutAnyOrigin], prefab_id: UInt32, net_id: Int) -> UInt64:
    _ = net_id
    ref world = world_ptr[unsafe_offset=0]
    var e = world.spawn()
    _ = world.insert[LocalTransform](e, LocalTransform(Vec3()))
    _ = world.insert[GlobalTransform](e, GlobalTransform())
    _ = world.insert[ReplicationComponent](e, ReplicationComponent(hitbox_radius=Float32(prefab_id)))
    return e.to_bits()


@fieldwise_init
struct _Health(NetPodComponent, Copyable, ImplicitlyCopyable, Movable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("tests.net_gaps.Health")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var hp: Int32
    var armor: UInt8

    def net_encode(self, mut writer: BitWriter):
        writer.write_u32(UInt32(Int(self.hp) & 0xFFFFFFFF))
        writer.write_bits(UInt64(self.armor), 4)

    @staticmethod
    def net_decode(mut reader: BitReader) -> Self:
        var hp = Int32(Int(reader.read_u32()))
        var armor = UInt8(reader.read_bits(4))
        return Self(hp, armor)


@fieldwise_init
struct _MoveInput(RollbackInput):
    var dx: Int8

    def __init__(out self):
        self.dx = 0

    def encode_bits(self, mut writer: BitWriter):
        writer.write_bits(UInt64(Int(self.dx) + 8), 4)

    def decode_bits(mut self, mut reader: BitReader):
        self.dx = Int8(Int(reader.read_bits(4)) - 8)

    def same_input(self, other: Self) -> Bool:
        return self.dx == other.dx


# ---------------------------------------------------------------------------
# 1. Server-spawned entities appear on the client (prefab hook) and despawn
# ---------------------------------------------------------------------------

def test_spawn_on_first_sight() raises:
    var sim = VirtualNetSim(seed=0x5EED01, one_way_latency_ms=5.0, jitter_ms=0.0, loss_rate=0.0, dup_rate=0.0, mtu_bytes=IPV6_SAFE_MTU_BYTES)
    var srv_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0, 9500), True)
    var srv_ep = srv_hub.local_endpoint()
    var cli_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(1, 9500), False)
    var cid = cli_hub.connect(srv_ep)
    assert_true(cli_hub.send_unreliable(cid, List[UInt8](), sim.now_ms()))
    var server = _make_app("SpawnSrv")
    _attach(server, srv_hub^)
    var client = _make_app("SpawnCli")
    _attach(client, cli_hub^)
    client.resources.get_ptr_mut[NetReplicationResource]().value()[unsafe_offset=0].register_spawn_hook(7, _spawn_prefab7)

    var e5 = _spawn_server_entity(server, 5, Vec3(3.0, 1.0, -2.0), prefab=7)
    _ = _spawn_server_entity(server, 6, Vec3(-4.0, 0.5, 8.0))
    var dt = Float32(1.0 / 60.0)
    for _ in range(20):
        server.step_one_frame(dt)
        sim.advance_ms(Float64(dt) * 1000.0)
        client.step_one_frame(dt)

    var f5 = _find_net_entity(client, 5)
    var f6 = _find_net_entity(client, 6)
    assert_true(f5[0] and f6[0], "client spawns both server entities on first sight")
    var nc5 = client.world.get[NetworkIdentityComponent](f5[1])
    assert_equal(Int(nc5.prefab_id), 7)
    assert_true(abs(client.world.get[ReplicationComponent](f5[1]).hitbox_radius - 7.0) < 1.0e-6, "prefab 7 hook built the entity")
    var p5 = client.world.get[LocalTransform](f5[1]).translation
    assert_true(abs(p5.x - 3.0) < 0.01 and abs(p5.y - 1.0) < 0.01 and abs(p5.z + 2.0) < 0.01)
    var p6 = client.world.get[LocalTransform](f6[1]).translation
    assert_true(abs(p6.x + 4.0) < 0.01 and abs(p6.z - 8.0) < 0.01, "default spawn hook places prefab-0 entity")
    assert_equal(client.resources.get_ptr[NetConfigResource]().value()[unsafe_offset=0].local_peer_id, 1)
    assert_equal(client.resources.get_ptr[NetReplicationResource]().value()[unsafe_offset=0].spawned_remote, 2)

    assert_true(server.world.despawn(e5))
    for _ in range(10):
        server.step_one_frame(dt)
        sim.advance_ms(Float64(dt) * 1000.0)
        client.step_one_frame(dt)
    assert_true(not client.world.is_alive(f5[1]), "server despawn removes the client-spawned entity")
    assert_true(client.world.is_alive(f6[1]))
    assert_true(not _find_net_entity(client, 5)[0])
    client.shutdown()
    server.shutdown()
    print("[PASS] server-spawned entities appear on the client via prefab hook, despawn follows")


# ---------------------------------------------------------------------------
# 2. Interpolated remote positions are smooth under jitter
# ---------------------------------------------------------------------------

def test_interpolation_smooth() raises:
    var sim = VirtualNetSim(seed=0x17E2, one_way_latency_ms=30.0, jitter_ms=12.0, loss_rate=0.0, dup_rate=0.0, mtu_bytes=IPV6_SAFE_MTU_BYTES)
    var srv_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0, 9510), True)
    var srv_ep = srv_hub.local_endpoint()
    var cli_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(1, 9510), False)
    var cid = cli_hub.connect(srv_ep)
    assert_true(cli_hub.send_unreliable(cid, List[UInt8](), sim.now_ms()))
    var server = _make_app("InterpSrv")
    _attach(server, srv_hub^)
    var client = _make_app("InterpCli")
    _attach(client, cli_hub^)
    client.resources.get_ptr_mut[NetConfigResource]().value()[unsafe_offset=0].interp_delay_ms = 100.0

    var se = _spawn_server_entity(server, 3, Vec3(0.0, 0.0, 0.0))
    var dt = Float32(1.0 / 60.0)
    var speed = Float32(3.0)
    var prev_x = Float32(0.0)
    var have_prev = False
    var max_step = Float32(0.0)
    var min_step = Float32(1.0e9)
    var raw_max_step = Float32(0.0)
    var prev_raw = Float32(0.0)
    var last_lag = Float32(0.0)
    for f in range(150):
        server.world.get_ptr_mut[LocalTransform](se).value()[unsafe_offset=0].translation = Vec3(speed * dt * Float32(f), 0.0, 0.0)
        server.step_one_frame(dt)
        sim.advance_ms(Float64(dt) * 1000.0)
        client.step_one_frame(dt)
        var fc = _find_net_entity(client, 3)
        if not fc[0]:
            continue
        var x = client.world.get[LocalTransform](fc[1]).translation.x
        var raw = client.resources.get_ptr[NetReplicationResource]().value()[unsafe_offset=0].receiver.entities[3].pos.x
        if f >= 40:
            if have_prev:
                var st = x - prev_x
                max_step = max(max_step, st)
                min_step = min(min_step, st)
                raw_max_step = max(raw_max_step, raw - prev_raw)
            have_prev = True
            last_lag = speed * dt * Float32(f) - x
        prev_x = x
        prev_raw = raw
    var ideal = speed * dt
    assert_true(min_step >= Float32(0.0), "interpolated motion never moves backwards")
    assert_true(max_step <= ideal * 2.5, "no snapping: per-frame step stays near the ideal 5 cm")
    assert_true(raw_max_step > max_step, "raw jittered snapshots are burstier than the interpolated pose")
    # Rendered ~interp_delay (100 ms) + ~one-way latency behind the server
    assert_true(last_lag > speed * 0.09 and last_lag < speed * 0.2, "interpolation runs at the configured delay")
    client.shutdown()
    server.shutdown()
    print("[PASS] interpolated remote positions are smooth (step min/max/ideal: " + String(min_step) + "/" + String(max_step) + "/" + String(ideal) + ")")


# ---------------------------------------------------------------------------
# 3. Client input upstream + prediction converges after loss and a server-side correction
# ---------------------------------------------------------------------------

def test_prediction_converges_after_loss() raises:
    var sim = VirtualNetSim(seed=0x9E3D, one_way_latency_ms=20.0, jitter_ms=0.0, loss_rate=0.0, dup_rate=0.0, mtu_bytes=IPV6_SAFE_MTU_BYTES)
    var srv_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0, 9520), True)
    var srv_ep = srv_hub.local_endpoint()
    var cli_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(1, 9520), False)
    var cid = cli_hub.connect(srv_ep)
    assert_true(cli_hub.send_unreliable(cid, List[UInt8](), sim.now_ms()))
    var server = _make_app("PredSrv")
    _attach(server, srv_hub^)
    var client = _make_app("PredCli")
    _attach(client, cli_hub^)
    client.resources.get_ptr_mut[NetConfigResource]().value()[unsafe_offset=0].predict_owned_entities = True

    var se = _spawn_server_entity(server, 9, Vec3(0.0, 0.0, 0.0), owner=1)
    var pred = PredictionResource[_MoveInput](capacity=128)
    var dt = Float32(1.0 / 60.0)
    var tick = 0
    var server_ticks = List[Int]()
    var wall = Float32(2.0)
    for f in range(160):
        # Server: consume client inputs (authoritative move with a wall the client does not model)
        ref srep = server.resources.get_ptr_mut[NetReplicationResource]().value()[unsafe_offset=0]
        while True:
            var inp = pop_client_input_as[_MoveInput](srep, 0)
            if not inp[0]:
                break
            server_ticks.append(inp[1])
            ref lt = server.world.get_ptr_mut[LocalTransform](se).value()[unsafe_offset=0]
            lt.translation.x = min(wall, lt.translation.x + Float32(Int(inp[2].dx)) * Float32(0.1))
        server.step_one_frame(dt)
        sim.advance_ms(Float64(dt) * 1000.0)
        if f == 25 or f == 45:
            sim.drop_next_outbound_packets(1, 4)  # lose 4 consecutive client datagrams
        # Client: predict locally once the owned entity exists
        var fc = _find_net_entity(client, 9)
        ref crep = client.resources.get_ptr_mut[NetReplicationResource]().value()[unsafe_offset=0]
        if fc[0] and client.world.get[NetworkIdentityComponent](fc[1]).is_local:
            tick += 1
            var mi = _MoveInput(Int8(1) if tick <= 40 else Int8(0))
            var cur = pred.latest()
            var nxt = PredictedState(cur.pos + Vec3(Float32(Int(mi.dx)) * Float32(0.1), 0.0, 0.0))
            assert_true(pred.predict_local(crep, tick, mi, nxt, dt))
        client.step_one_frame(dt)
        ref crep2 = client.resources.get_ptr_mut[NetReplicationResource]().value()[unsafe_offset=0]
        var rr = pred.reconcile_from_server(crep2, 9)
        if rr[0] and fc[0]:
            client.world.get_ptr_mut[LocalTransform](fc[1]).value()[unsafe_offset=0].translation = rr[1].pos
    var srv_x = server.world.get[LocalTransform](se).translation.x
    var cli_x = pred.latest().pos.x
    assert_true(abs(srv_x - wall) < 1.0e-4, "server applied the authoritative wall clamp")
    assert_true(abs(cli_x - srv_x) < 0.02, "predicted client converges onto the server state")
    assert_true(pred.buffer.mispredictions > 0, "wall correction was detected as a misprediction")
    # Redundant input frames: the first 40 movement ticks all reached the server despite loss
    var seen = 0
    for i in range(len(server_ticks)):
        if server_ticks[i] >= 1 and server_ticks[i] <= 40:
            seen += 1
    assert_equal(seen, 40)
    for i in range(1, len(server_ticks)):
        assert_true(server_ticks[i] > server_ticks[i - 1], "server consumes input ticks in order, once")
    var fc_end = _find_net_entity(client, 9)
    assert_true(fc_end[0])
    assert_true(abs(client.world.get[LocalTransform](fc_end[1]).translation.x - srv_x) < 0.02, "snapshots never overwrite the predicted entity with stale poses")
    client.shutdown()
    server.shutdown()
    print("[PASS] tick-stamped redundant input upstream; predicted client converges after loss (srv " + String(srv_x) + ", cli " + String(cli_x) + ")")


# ---------------------------------------------------------------------------
# 4. RPCs delivered exactly once; unauthorised writes / RPCs rejected; POD component replication
# ---------------------------------------------------------------------------

def test_rpc_authority_and_pod() raises:
    var sim = VirtualNetSim(seed=0xA11CE, one_way_latency_ms=15.0, jitter_ms=4.0, loss_rate=0.15, dup_rate=0.05, mtu_bytes=IPV6_SAFE_MTU_BYTES)
    var srv_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0, 9530), True)
    var srv_ep = srv_hub.local_endpoint()
    var cli_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(1, 9530), False)
    var cid = cli_hub.connect(srv_ep)
    for _ in range(4):
        assert_true(cli_hub.send_unreliable(cid, List[UInt8](), sim.now_ms()))
    var server = _make_app("RpcSrv")
    _attach(server, srv_hub^)
    var client = _make_app("RpcCli")
    _attach(client, cli_hub^)
    _ = server.world.register_component[_Health]()
    _ = client.world.register_component[_Health]()
    assert_true(register_net_pod_component[_Health](server.resources.get_ptr_mut[NetReplicationResource]().value()[unsafe_offset=0], 0))
    assert_true(register_net_pod_component[_Health](client.resources.get_ptr_mut[NetReplicationResource]().value()[unsafe_offset=0], 0))

    var ea = _spawn_server_entity(server, 20, Vec3(), owner=1, authority=NET_AUTHORITY_CLIENT)
    var eb = _spawn_server_entity(server, 21, Vec3(1.0, 0.0, 1.0))
    var ec = _spawn_server_entity(server, 22, Vec3(-1.0, 0.0, 2.0))
    _ = server.world.insert[_Health](ec, _Health(100, 3))
    var dt = Float32(1.0 / 60.0)
    var srv_rpc_counts = List[Int](length=8, fill=0)
    var cli_rpc_counts = List[Int](length=16, fill=0)
    var rpcs_queued = False
    var tampered = False
    var health_changed = False
    for f in range(180):
        server.step_one_frame(dt)
        sim.advance_ms(Float64(dt) * 1000.0)
        client.step_one_frame(dt)
        var fa = _find_net_entity(client, 20)
        var fb = _find_net_entity(client, 21)
        ref crep = client.resources.get_ptr_mut[NetReplicationResource]().value()[unsafe_offset=0]
        ref srep = server.resources.get_ptr_mut[NetReplicationResource]().value()[unsafe_offset=0]
        if fa[0] and fb[0] and client.resources.get_ptr[NetConfigResource]().value()[unsafe_offset=0].local_peer_id == 1:
            if not rpcs_queued:
                var pl = List[UInt8]()
                pl.append(42)
                crep.send_rpc(0, 1, 20, pl)  # owned target: allowed
                crep.send_rpc(0, 2, 21, pl)  # not owned: rejected
                crep.send_rpc(0, 3, NET_RPC_TARGET_NONE, pl)  # global: allowed
                srep.send_rpc(NET_PEER_ALL, 7, NET_RPC_TARGET_NONE, pl)
                srep.send_rpc(1, 8, 21, pl)
                srep.send_rpc(2, 9, NET_RPC_TARGET_NONE, pl)  # nobody is peer 2
                rpcs_queued = True
            # Legit client-authoritative write on the owned entity
            client.world.get_ptr_mut[LocalTransform](fa[1]).value()[unsafe_offset=0].translation = Vec3(4.0, 0.0, 0.0)
            # Malicious client: claims ownership/authority of a server entity locally
            if not tampered:
                ref ncb = client.world.get_ptr_mut[NetworkIdentityComponent](fb[1]).value()[unsafe_offset=0]
                ncb.owner_peer_id = 1
                ncb.authority = NET_AUTHORITY_CLIENT
                ncb.is_local = True
                tampered = True
            client.world.get_ptr_mut[LocalTransform](fb[1]).value()[unsafe_offset=0].translation = Vec3(9.0, 9.0, 9.0)
        if f == 120:
            server.world.get_ptr_mut[_Health](ec).value()[unsafe_offset=0].hp = 55
            health_changed = True
        var srv_in = srep.drain_rpcs()
        for i in range(len(srv_in)):
            assert_equal(srv_in[i].from_peer, 1)
            assert_equal(Int(srv_in[i].payload[0]), 42)
            srv_rpc_counts[Int(srv_in[i].rpc_id)] += 1
        var cli_in = crep.drain_rpcs()
        for i in range(len(cli_in)):
            assert_equal(cli_in[i].from_peer, 0)
            cli_rpc_counts[Int(cli_in[i].rpc_id)] += 1

    ref srep_end = server.resources.get_ptr[NetReplicationResource]().value()[unsafe_offset=0]
    assert_true(rpcs_queued)
    assert_equal(srv_rpc_counts[1], 1)
    assert_equal(srv_rpc_counts[3], 1)
    assert_equal(srv_rpc_counts[2], 0)
    assert_equal(srep_end.rejected_rpcs, 1)
    assert_equal(cli_rpc_counts[7], 1)
    assert_equal(cli_rpc_counts[8], 1)
    assert_equal(cli_rpc_counts[9], 0)
    print("[PASS] reliable RPCs delivered exactly once under 15% loss + 5% duplication; non-owner RPC rejected")

    var pa = server.world.get[LocalTransform](ea).translation
    var pb = server.world.get[LocalTransform](eb).translation
    assert_true(abs(pa.x - 4.0) < 1.0e-4, "owner's client-authoritative write accepted")
    assert_true(srep_end.accepted_writes > 0)
    assert_true(abs(pb.x - 1.0) < 1.0e-6 and abs(pb.y) < 1.0e-6, "unauthorised client write rejected")
    assert_true(srep_end.rejected_writes > 0)
    print("[PASS] client-authoritative writes gated by owner_peer_id/authority (accepted " + String(srep_end.accepted_writes) + ", rejected " + String(srep_end.rejected_writes) + ")")

    assert_true(health_changed)
    var fc_end = _find_net_entity(client, 22)
    assert_true(fc_end[0])
    var h = client.world.get_ptr[_Health](fc_end[1])
    assert_true(Bool(h), "POD component replicated onto the client entity")
    assert_equal(Int(h.value()[unsafe_offset=0].hp), 55)
    assert_equal(Int(h.value()[unsafe_offset=0].armor), 3)
    client.shutdown()
    server.shutdown()
    print("[PASS] extra POD component payload replicated and delta-updated")


# ---------------------------------------------------------------------------
# 5. X25519 + HKDF session keys inside the cookie handshake, connect-token admission
# ---------------------------------------------------------------------------

def _pump(mut sim: VirtualNetSim, mut srv: NetHub[VirtualEndpoint], mut cli: NetHub[VirtualEndpoint], frames: Int):
    for _ in range(frames):
        var now = sim.now_ms()
        cli.poll(now)
        srv.poll(now)
        sim.advance_ms(10.0)


def test_secure_handshake() raises:
    var sim = VirtualNetSim(seed=0xC0FFEE, one_way_latency_ms=10.0, jitter_ms=0.0, loss_rate=0.0)
    var srv = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0, 9540), True)
    var good = NetHub[VirtualEndpoint](sim.create_ipv4_peer(1, 9540), False)
    var bad = NetHub[VirtualEndpoint](sim.create_ipv4_peer(2, 9540), False)
    var none = NetHub[VirtualEndpoint](sim.create_ipv4_peer(3, 9540), False)
    var srv_ep = srv.local_endpoint()
    var server_priv = _hex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb")
    assert_true(srv.enable_secure_handshake(AeadKey.from_u64x4(1, 2, 3, 4), server_priv))
    var token_key = _bytes_of("matchmaker-hmac-key")
    srv.require_connect_tokens(token_key, 0x464C)

    var tok = issue_connect_token(token_key, 0x464C, 777, 60000)
    assert_true(good.begin_secure_connect(srv_ep, _hex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a"), tok))
    _pump(sim, srv, good, 10)
    assert_true(good.kx_established, "client derived session keys from the server cookie")
    assert_equal(srv.open_count(), 1)
    assert_equal(Int(srv.peer_client_id(0)), 777)
    var ccid = good.kx_cid
    # Per-direction keys match crosswise and differ per direction
    assert_true(srv.chans[0].aead_key.w0 == good.chans[ccid].aead_rx_key.w0)
    assert_true(srv.chans[0].aead_rx_key.w3 == good.chans[ccid].aead_key.w3)
    assert_true(srv.chans[0].aead_key.w0 != srv.chans[0].aead_rx_key.w0)
    assert_true(good.send_reliable(ccid, _bytes_of("hello server")))
    assert_true(srv.send_reliable(0, _bytes_of("hello client")))
    var got_s = False
    var got_c = False
    for _ in range(20):
        _pump(sim, srv, good, 1)
        var m = srv.pop_reliable(0)
        if m[0]:
            got_s = _eq(m[1], _bytes_of("hello server"))
        var m2 = good.pop_reliable(ccid)
        if m2[0]:
            got_c = _eq(m2[1], _bytes_of("hello client"))
        if got_s and got_c:
            break
    assert_true(got_s and got_c, "encrypted traffic flows both ways with X25519/HKDF session keys")
    assert_equal(srv.chans[0].crypto_auth_failures, 0)

    var forged = tok.copy()
    forged[16] ^= 0xFF  # tamper with expiry
    assert_true(bad.begin_secure_connect(srv_ep, _hex("4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d"), forged))
    _pump(sim, srv, bad, 10)
    assert_true(none.begin_secure_connect(srv_ep, _hex("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4")))
    _pump(sim, srv, none, 10)
    assert_equal(srv.open_count(), 1)
    assert_true(srv.tokens_rejected >= 2, "forged and missing connect tokens rejected")
    print("[PASS] cookie handshake + X25519/HKDF session keys + HMAC connect-token admission (rejected " + String(srv.tokens_rejected) + ")")


def main() raises:
    test_kex_vectors()
    test_secure_handshake()
    test_spawn_on_first_sight()
    test_interpolation_smooth()
    test_prediction_converges_after_loss()
    test_rpc_authority_and_pod()
